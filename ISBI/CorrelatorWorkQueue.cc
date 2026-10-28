#include "Common/Config.h"

#include "Common/BandPass.h"
#include "ISBI/CorrelatorWorkQueue.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <complex>


CorrelatorWorkQueue::CorrelatorWorkQueue(ISBI_CorrelatorPipeline &pipeline, DeviceInstance &deviceInstance)
:
  ps(static_cast<const ISBI_Parset &>(pipeline.ps)),
  pipeline(pipeline),
  deviceInstance(deviceInstance),

  hostDelays(boost::extents[ps.nrStations()][2]),

  validData(ps.inputDescriptors().size()) // FIXME???

#if defined USE_SEPARATE_THREAD
, stop(false),
  bufferEmpty(1),
  bufferFull(0)
#endif
{
  memset(hostDelays.origin(), 0, hostDelays.bytesize());
}


bool CorrelatorWorkQueue::hasValidData(std::vector<SparseSet<TimeStamp> > &validData)
{
  for (SparseSet<TimeStamp> &set : validData)
    if (!set.empty())
      return true;

  return false;
}


void CorrelatorWorkQueue::doWork()
{
#if !defined CL_DEVICE_TOPOLOGY_AMD
#pragma omp for schedule(dynamic), collapse(2), ordered // nowait
  for (TimeStamp time = ps.startTime(); time < ps.stopTime(); time += ps.nrSamplesPerSubbandBeforeFilter())
    for (unsigned subband = 0; subband < ps.nrSubbands(); subband ++)
      doSubband(time, subband);
#else
  TimeStamp time;
  unsigned  subband;

  while (pipeline.getWork(deviceInstance.numaNode, time, subband))
    doSubband(time, subband);
#endif
}


bool CorrelatorWorkQueue::inTime(const TimeStamp &time)
{
  return !ps.realTime() || (int64_t) TimeStamp::now(ps.clockSpeed()) - (int64_t) time < ps.nrRingBufferSamplesPerSubband() - ps.nrSamplesPerSubbandBeforeFilter();
}


void CorrelatorWorkQueue::computeWeights(const std::vector<SparseSet<TimeStamp> > &validData, Visibilities *visibilities)
{
  for (unsigned stat2 = 0, pair = 0; stat2 < validData.size(); stat2 ++)
    for (unsigned stat1 = 0; stat1 <= stat2 && pair < sizeof(visibilities->header.weights) / sizeof(visibilities->header.weights[0]); stat1 ++, pair ++)
      visibilities->header.weights[pair] = ps.producesBaseline(stat1, stat2) ?
        ((uint32_t) (int64_t) (validData[stat1] & validData[stat2]).count() / ps.nrChannelsPerSubbandBeforeFilter() - (NR_TAPS - 1)) * ps.channelIntegrationFactor() :
        0;
}


// With --referenceStation, only baselines involving that station are produced.
// The Tensor-Core Correlator has no notion of a baseline subset, so it still
// computes them all -- the cost of that is negligible, since it scales with
// nrStations^2 while the filter and the host-to-device copy, which dominate,
// scale linearly. What this does buy is correctness of interpretation: once the
// array is mixed-mode, only the reference station's grid is common to everyone,
// so a pair that excludes it is not a meaningful correlation and must not be
// mistaken for one. Zeroing the data AND the weight says "not computed" in a
// way every existing reader already understands, because a zero weight is how a
// fully flagged baseline has always been reported.
void CorrelatorWorkQueue::zeroUnproducedBaselines(Visibilities *visibilities)
{
  if (ps.referenceStation() < 0)
    return;

  for (unsigned stat2 = 0, pair = 0; stat2 < ps.nrStations(); stat2 ++)
    for (unsigned stat1 = 0; stat1 <= stat2; stat1 ++, pair ++)
      if (!ps.producesBaseline(stat1, stat2))
        for (unsigned channel = 0; channel < ps.nrOutputChannelsPerSubband(); channel ++)
          for (unsigned pol0 = 0; pol0 < ps.nrPolarizations(); pol0 ++)
            for (unsigned pol1 = 0; pol1 < ps.nrPolarizations(); pol1 ++)
              visibilities->hostVisibilities[channel][pair][pol0][pol1] = 0.0f;
}


// The integer sample delay is applied by shifting the sampled data (see
// InputSection::enqueueHostToDeviceCopy). For a signal that was mixed down
// before sampling, a delay of n samples on the sky also multiplies the
// baseband signal by exp(2 pi i f_LO n / Fs); a plain sample shift supplies the
// baseband part of the delay but not that LO phase. The filter kernel applies
// the fractional delay as a rotation at the *sky* frequency, so the fractional
// part does get it; the integer part does not, and every change of n would
// otherwise jump the baseline phase by 2 pi f_LO / Fs per sample. The missing
// factor is channel independent, so restore it here as one complex factor per
// baseline per block (-125 deg per sample at 6.6 GHz / 32 MS/s). f_LO is the
// sky frequency at baseband DC: the lower band edge, or the upper one for
// subbands whose spectrum the filter mirrors. Sign and edge were verified on
// scan E011/No0017, whose delay crosses n+1/2 mid-scan: with this factor the
// phase is continuous through the crossing on all 8 subbands; without it the
// jump is -125 deg, with the opposite sign +115 deg.
void CorrelatorWorkQueue::correctIntegerDelayPhase(unsigned subband, const std::vector<int64_t> &integerStationDelays, Visibilities *visibilities)
{
  bool   mirrored    = (subband & 1) == 0; // DeviceInstance::filterIndexForSubband: even subbands use the mirrored filter
  double loFrequency = ps.subbandFrequencies()[subband] + (mirrored ? 0.5 : -0.5) * ps.subbandBandwidth();
  double Fs          = ps.sampleRate();

  for (unsigned stat2 = 0, pair = 0; stat2 < ps.nrStations(); stat2 ++)
    for (unsigned stat1 = 0; stat1 <= stat2; stat1 ++, pair ++) {
      int64_t dn = integerStationDelays[stat2] - integerStationDelays[stat1];

      if (dn == 0)
        continue;

      double turns = std::fmod(loFrequency / Fs * dn, 1.0);
      std::complex<float> factor = std::polar(1.0f, static_cast<float>(-2 * M_PI * turns));

      for (unsigned channel = 0; channel < ps.nrOutputChannelsPerSubband(); channel ++)
        for (unsigned pol0 = 0; pol0 < ps.nrPolarizations(); pol0 ++)
          for (unsigned pol1 = 0; pol1 < ps.nrPolarizations(); pol1 ++)
            visibilities->hostVisibilities[channel][pair][pol0][pol1] *= factor;
    }
}


void CorrelatorWorkQueue::doSubband(const TimeStamp &time, unsigned subband)
{
  pipeline.startReadTransaction(time);
  pipeline.inputSection.fillInMissingSamples(time, subband, validData);

  if (hasValidData(validData) && inTime(time)) {
    std::unique_ptr<Visibilities> visibilities = pipeline.outputSection.getVisibilitiesBuffer(subband);
    std::vector<int64_t> integerStationDelays(ps.nrStations(), 0);

    // TODO:
    // if (pipeline.delayCorrection) {
    std::vector<DelayCorrection::StationDelay> stationDelays = pipeline.delayCorrection.stationDelays(time);

    if (stationDelays.size() != ps.nrStations()) {
      throw std::runtime_error("unexpected amount of delays");
    }

    for (unsigned station = 0; station < ps.nrStations(); ++station) {
      integerStationDelays[station] = stationDelays[station].integerSamples;
      hostDelays[station][0] = stationDelays[station].d0;
      hostDelays[station][1] = stationDelays[station].d1;
    }

    // } else set delays to 0
    
    std::function<void (cu::Stream &, cu::DeviceMemory &, PerformanceCounter &)> enqueueCopyInputBuffer = [=] (cu::Stream &stream, cu::DeviceMemory &devInputBuffer, PerformanceCounter &counter)
    {
      pipeline.inputSection.enqueueHostToDeviceCopy(stream, devInputBuffer, counter, time, subband, integerStationDelays);
    };

    unsigned nrHistorySamples = (NR_TAPS - 1) * ps.nrChannelsPerSubbandBeforeFilter();
    unsigned startIndex = (time - nrHistorySamples) % ps.nrRingBufferSamplesPerSubband();
    deviceInstance.doSubband(time, subband, enqueueCopyInputBuffer, pipeline.inputSection.hostRingBuffers[subband], hostDelays, visibilities->hostVisibilities, startIndex);

    if (ps.delayCompensation())
      correctIntegerDelayPhase(subband, integerStationDelays, visibilities.get());

    zeroUnproducedBaselines(visibilities.get());

    visibilities->startTime = time;
    visibilities->endTime = time + ps.nrSamplesPerSubbandBeforeFilter();
    computeWeights(validData, visibilities.get());
    pipeline.outputSection.putVisibilitiesBuffer(std::move(visibilities), time, subband);
  } else {
    if (subband == 0)
#pragma omp critical (clog)
      std::clog << "Warning: no valid samples for block starting at " << time << std::endl;

    pipeline.outputSection.putVisibilitiesBuffer(nullptr, time, subband);
  }

  pipeline.endReadTransaction(time);
}

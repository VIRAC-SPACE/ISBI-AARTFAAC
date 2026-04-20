#include "Common/Config.h"

#include "Common/BandPass.h"
#include "ISBI/CorrelatorWorkQueue.h"

#include <iostream>


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
  memset(hostDelays.origin(), 9, hostDelays.bytesize());
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
      visibilities->header.weights[pair] = ((uint32_t) (int64_t) (validData[stat1] & validData[stat2]).count() / ps.nrChannelsPerSubbandBeforeFilter() - (NR_TAPS - 1)) * ps.channelIntegrationFactor();
}


void CorrelatorWorkQueue::doSubband(const TimeStamp &time, unsigned subband)
{
  pipeline.startReadTransaction(time);
  pipeline.inputSection.fillInMissingSamples(time, subband, validData);

  if (hasValidData(validData) && inTime(time)) {
    std::unique_ptr<Visibilities> visibilities = pipeline.outputSection.getVisibilitiesBuffer(subband);
    std::vector<int64_t> integerStationDelays(ps.nrStations, 0);

    // TODO:
    // if (pipeline.delayCorrection) {
    std::vector<DelayCorrection::StationDelays> stationDelays = pipeline.delayCorrection.stationDelays(time);

    if (stationDelays.size() != ps.nrStations()) {
      throw std::runtime_error("unexpected amount of delays");
    }

    for (unsigned station = 0; station < ps.nrStations(); ++station) {
      integerStationDelay[station] = stationDelays[station].integerSamples;
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

#include "Common/Config.h"

#include "Common/Affinity.h"
#include "ISBI/InputBuffer.h"
#include "ISBI/InputSection.h"

#include <fstream>
#include <map>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace {
   auto getDelayAt (const std::map<int64_t, double>& delays, int64_t timestamp) {
    if (delays.empty()) {
      throw std::runtime_error("Delay map is empty");
    }

    // Exact match
    auto exact = delays.find(timestamp);
    if (exact != delays.end()) {
      return exact->second;
    }

    // Need at least 2 points to interpolate at all
    if (delays.size() < 2) {
      throw std::runtime_error("Not enough points to interpolate");
    }

    // Out of range check
    if (timestamp < delays.begin()->first || timestamp > delays.rbegin()->first) {
      throw std::runtime_error("Timestamp outside interpolation range");
    }

    // Flatten map into vectors for indexed access
    std::vector<double> x;
    std::vector<double> y;
    x.reserve(delays.size());
    y.reserve(delays.size());

    for (const auto& [ts, val] : delays) {
      x.push_back(static_cast<double>(ts));
      y.push_back(val);
    }

    const int n = static_cast<int>(x.size());

    // Find interval [x[i], x[i+1]] containing timestamp
    auto upper = std::upper_bound(x.begin(), x.end(), static_cast<double>(timestamp));
    int i = static_cast<int>(std::distance(x.begin(), upper)) - 1;

    // Safety clamp
    i = std::max(0, std::min(i, n - 2));

    const double xt = static_cast<double>(timestamp);

    // Fallback to linear interpolation if not enough points for Akima
    if (n < 5) {
      double t = (xt - x[i]) / (x[i + 1] - x[i]);
      return y[i] + t * (y[i + 1] - y[i]);
    }

    // Slopes between consecutive points
    std::vector<double> m(n - 1);
    for (int j = 0; j < n - 1; ++j) {
      m[j] = (y[j + 1] - y[j]) / (x[j + 1] - x[j]);
    }

    // Derivatives at each point
    std::vector<double> d(n);

    auto akimaDerivative = [&](int k) -> double {
      // For edges, fall back to simpler estimates
      if (k < 2 || k > n - 3) {
        if (k == 0) return m[0];
        if (k == 1) return 0.5 * (m[0] + m[1]);
        if (k == n - 2) return 0.5 * (m[n - 3] + m[n - 2]);
        if (k == n - 1) return m[n - 2];
      }

      double w1 = std::abs(m[k + 1] - m[k]);
      double w2 = std::abs(m[k - 1] - m[k - 2]);

      if (w1 + w2 == 0.0) {
        return 0.5 * (m[k - 1] + m[k]);
      }

      return (w1 * m[k - 1] + w2 * m[k]) / (w1 + w2);
    };

    for (int k = 0; k < n; ++k) {
      d[k] = akimaDerivative(k);
    }

    // Cubic Hermite interpolation on interval [i, i+1]
    double h = x[i + 1] - x[i];
    double t = (xt - x[i]) / h;

    double h00 =  2.0 * t * t * t - 3.0 * t * t + 1.0;
    double h10 =        t * t * t - 2.0 * t * t + t;
    double h01 = -2.0 * t * t * t + 3.0 * t * t;
    double h11 =        t * t * t -       t * t;

    return h00 * y[i]
      + h10 * h * d[i]
      + h01 * y[i + 1]
      + h11 * h * d[i + 1];
  };
}

InputSection::InputSection(const ISBI_Parset &ps)
:
  ps(ps),

  hostRingBuffers([&] () {
    std::vector<MultiArrayHostBuffer<char, 4>> buffers; 

    for (unsigned subband = 0; subband < ps.nrSubbands(); subband ++)
        buffers.emplace_back(std::move(boost::extents[ps.nrStations()][ps.nrPolarizations()][ps.nrRingBufferSamplesPerSubband()][ps.nrBytesPerRealSample()]), CU_MEMHOSTALLOC_WRITECOMBINED);

    return std::move(buffers);
  } ()),

  inputBuffers([&] () {
    std::vector<std::unique_ptr<InputBuffer>> buffers;

    for (unsigned stationSet = 0; stationSet < ps.inputDescriptors().size(); stationSet ++) {
      std::unique_ptr<BoundThread> bt(ps.inputBufferNodes().size() > 0 ? new BoundThread(ps.allowedCPUs(ps.inputBufferNodes()[stationSet])) : nullptr);
      buffers.emplace_back(new InputBuffer(ps, &hostRingBuffers[0], 0, ps.nrSubbands(), stationSet, 1, nrTimesPerPacket));
    }

    return std::move(buffers);
  } ())
{
}



InputSection::~InputSection()
{
#pragma omp parallel for
  for (unsigned i = 0; i < inputBuffers.size(); i ++)
    inputBuffers[i] = nullptr;
}




void InputSection::enqueueHostToDeviceCopy(cu::Stream &stream, cu::DeviceMemory &devBuffer, PerformanceCounter &counter, const TimeStamp &startTime, unsigned subband) {
  int referenceStation = 0;

  for (unsigned station = 0; station < ps.nrStations(); station++) {
    double delayAtStart  = getDelayAt(ps.delays()[station], (int64_t)startTime);

    if (station != referenceStation) {
      double delayAtStartR = getDelayAt(ps.delays()[referenceStation], (int64_t)startTime);
      delayAtStart -= delayAtStartR;
    } else {
      delayAtStart = 0.0;
    }

    double Fs = (double)ps.sampleRate();
    double delayInSamples = delayAtStart * Fs;
    int delay = static_cast<int>(std::llround(delayInSamples));

    if (subband == 0) {
      std::cout << "InputSection" << std::endl;
      std::cout << "time=" << (int64_t)startTime <<  " delay=" << delay << std::endl;
    }

    unsigned nrHistorySamples = (NR_TAPS - 1) * ps.nrChannelsPerSubbandBeforeFilter();

    TimeStamp earlyStartTime   = startTime - nrHistorySamples + delay;
    TimeStamp endTime          = startTime + ps.nrSamplesPerSubbandBeforeFilter() + delay;

    unsigned startTimeIndex = earlyStartTime % ps.nrRingBufferSamplesPerSubband();
    unsigned endTimeIndex = endTime % ps.nrRingBufferSamplesPerSubband();

    unsigned nrBytesPerTime = ps.nrBytesPerRealSample();

    {
      PerformanceCounter::Measurement measurement(counter, stream, 0, 0, (endTime - earlyStartTime) * nrBytesPerTime);

      uint32_t n = endTime - earlyStartTime;
      assert(n <= ps.nrRingBufferSamplesPerSubband());

      for (unsigned pol = 0; pol < ps.nrPolarizations(); pol++) {
        size_t offset = (station * ps.nrPolarizations() + pol) * n * nrBytesPerTime;

        uint32_t firstPart = ps.nrRingBufferSamplesPerSubband() - startTimeIndex;
        uint32_t secondPart = 0; 

        if (startTimeIndex < endTimeIndex) {
          firstPart = endTimeIndex - startTimeIndex;
        } else {
          secondPart = n - firstPart;
        }

        assert(firstPart + secondPart == n);

        if (firstPart > 0) {
          cu::DeviceMemory dst(devBuffer + offset);
          stream.memcpyHtoDAsync(
              dst, 
              hostRingBuffers[subband][station][pol][startTimeIndex].origin(),
              firstPart * nrBytesPerTime
              );
        }

        if (secondPart > 0) {
          cu::DeviceMemory dst(devBuffer + offset + firstPart * nrBytesPerTime);
          stream.memcpyHtoDAsync(
              dst,
              hostRingBuffers[subband][station][pol][0].origin(),
              secondPart * nrBytesPerTime
              );
        }
      }
    }
  }
#if 0
  char filename[1024];
  sprintf(filename, "/var/scratch/romein/out.%u", subband + 8);
  std::ofstream file(filename, std::ios::binary | std::ios::app);

  for (TimeStamp time = startTime; time < endTime; time ++)
    file.write(hostRingBuffers[subband][time % ps.nrRingBufferSamplesPerSubband()].origin(), ps.nrStations() * ps.nrPolarizations() * ps.nrBytesPerComplexSample());
#endif
}



void InputSection::fillInMissingSamples(const TimeStamp &time, unsigned subband, std::vector<SparseSet<TimeStamp> > &validData)
{
  for (unsigned stationSet = 0; stationSet < inputBuffers.size(); stationSet ++)
    inputBuffers[stationSet]->fillInMissingSamples(time, subband, validData[stationSet]);
}


void InputSection::startReadTransaction(const TimeStamp &time)
{
  for (std::unique_ptr<InputBuffer> &inputBuffer : inputBuffers)
    inputBuffer->startReadTransaction(time);
}


void InputSection::endReadTransaction(const TimeStamp &time)
{
  for (std::unique_ptr<InputBuffer> &inputBuffer : inputBuffers)
    inputBuffer->endReadTransaction(time);
}


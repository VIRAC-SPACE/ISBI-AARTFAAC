#if !defined CORRELATOR_FILTER_H
#define CORRELATOR_FILTER_H

#include "Common/PerformanceCounter.h"
#include "Correlator/Parset.h"

#include <cudawrappers/cu.hpp>

#include <libfilter/Filter.h>
#include <optional>

class Filter {
  public:
    Filter(const cu::Device &, const CorrelatorParset &, bool mirror);

    void launchAsync(cu::Stream &stream,
                     cu::DeviceMemory &devCorrectedData,
                     const cu::DeviceMemory &devInSamples,
                     PerformanceCounter &counter,
                     const std::optional<const cu::DeviceMemory> &devDelays = std::nullopt,
                     const std::optional<double> subbandCenterFrequency = std::nullopt);

  private:
    const CorrelatorParset ps;
    tcc::Filter filter;
};

#endif

#include "Correlator/Filter.h"

Filter::Filter(const cu::Device &device, const CorrelatorParset &ps, bool mirror) :
  ps(ps),
  filter(device, tcc::FilterArgs {
      .nrReceivers                  = ps.nrStations(),
      .nrChannels                   = ps.nrChannelsPerSubband(),
      .nrSamplesPerChannel          = ps.nrSamplesPerChannel(),
      .nrPolarizations              = ps.nrPolarizations(),
      .input                        = tcc::FilterArgs::Input {
          .sampleFormat             = tcc::FilterArgs::Format::i8,
          .isPurelyReal             = true,
      },
      .firFilter                    = tcc::FilterArgs::FIR_Filter {
          .nrTaps                   = NR_TAPS,
          .sampleFormat             = tcc::FilterArgs::Format::fp32,
      },
      .fft                          = tcc::FilterArgs::FFT {
          .sampleFormat             = tcc::FilterArgs::Format::fp32,
          .shift                    = false,
          .mirror                   = mirror,
      },
      .delays                       = ps.delayCompensation() ? std::optional<tcc::FilterArgs::Delays>(tcc::FilterArgs::Delays {
          .subbandBandwidth         = ps.subbandBandwidth(),
          .polynomialOrder          = 1,
          .separatePerPolarization  = false,
      }) : std::nullopt,
      .bandPassCorrection           = std::nullopt,
      .output                       = tcc::FilterArgs::Output {
          .sampleFormat             = tcc::FilterArgs::Format::fp16,
      },
  }) {
}

void Filter::launchAsync(cu::Stream &stream,
                         cu::DeviceMemory &devOutSamples,
                         const cu::DeviceMemory &devInSamples,
                         PerformanceCounter &counter,
                         const std::optional<const cu::DeviceMemory> &devDelays,
                         const std::optional<double> subbandCenterFrequency) {
  PerformanceCounter::Measurement measurement(counter, stream, filter.nrOperations(), 0, 0);

  if (ps.delayCompensation()) {
    filter.launchAsync(stream, devOutSamples, devInSamples, devDelays, subbandCenterFrequency);
  } else {
    filter.launchAsync(stream, devOutSamples, devInSamples);
  }
}

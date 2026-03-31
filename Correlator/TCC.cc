#include "Correlator/TCC.h"

namespace {
  tcc::Format bitsToFormat(unsigned nrBits)
  {
    switch (nrBits) {
      case  4: return tcc::i4;
      case  8: return tcc::i8;
      case 16: return tcc::fp16;
      default:
        throw std::invalid_argument("nrBits should be 4, 8, or 16");
    }
  }
}

TCC::TCC(const cu::Device &device, const CorrelatorParset &ps)
:
  tcc(device,
      bitsToFormat(ps.nrBitsPerSample()),
      ps.nrStations(),
      ps.nrOutputChannelsPerSubband(),
      ps.nrSamplesPerChannel() * ps.channelIntegrationFactor(),
      ps.nrPolarizations(),
      "typedef float2 Visibilities[" + std::to_string(ps.nrBaselines()) + "]["
                                    + std::to_string(ps.nrOutputChannelsPerSubband()) + "]["
                                    + std::to_string(ps.nrPolarizations()) + "]["
                                    + std::to_string(ps.nrPolarizations()) + "];"
      "template <bool add> __device__ inline void storeVisibility(Visibilities visibilities, "
      "unsigned channel, unsigned recvY, unsigned recvX, unsigned polY, unsigned polX, float2 visibility)"
      "{"
        "unsigned baseline = recvY * (recvY + 1) / 2 + recvX;"
        "visibilities[baseline][channel][polY][polX] = make_float2(visibility.x, visibility.y);"
      "}")
{
  if (ps.nrPolarizations() != 2)
    throw Parset::Error("the Tensor-Core Correlator currently only supports 2 polarizations");

  if (ps.correlationMode() != 15)
    throw Parset::Error("the Tensor-Core Correlator currently only supports correlator mode 15");
}

void TCC::launchAsync(cu::Stream &stream, cu::DeviceMemory &devVisiblities, const cu::DeviceMemory &devCorrectedData, PerformanceCounter &counter)
{
  PerformanceCounter::Measurement measurement(counter, stream, tcc.FLOPS(), 0, 0);
  tcc.launchAsync(stream, devVisiblities, devCorrectedData, true);
}

#include "Common/Config.h"

#include "Common/CUDA_Support.h"
#include "Correlator/CorrelatorPipeline.h"
#include "Correlator/DeviceInstance.h"
#include "Correlator/Parset.h"

#include <cassert>
#include <iostream>


static void setTestPattern(const CorrelatorParset &ps, boost::multi_array_ref<char, 4> &inputSamples)
{
  if (ps.nrStations() >= 3)
  {
    double centerFrequency = 384 * ps.subbandBandwidth();
    double baseFrequency = centerFrequency - .5 * ps.subbandBandwidth();
    double testSignalChannel = ps.nrChannelsPerSubbandBeforeFilter() / 5.0;
    double signalFrequency = baseFrequency + testSignalChannel * ps.subbandBandwidth() / ps.nrChannelsPerSubbandBeforeFilter();

    assert(ps.nrChannelsPerSubbandBeforeFilter() > 1);// FIXME: 1 ch = not implemented
    for (unsigned time = 0; time < (NR_TAPS - 1 + ps.nrSamplesPerChannel()) * ps.nrChannelsPerSubbandBeforeFilter(); time ++)
    {
      double phi = 2.0 * M_PI * signalFrequency * time / ps.subbandBandwidth();

      switch (ps.nrBytesPerComplexSample())
      {
	case 4 :
          reinterpret_cast<std::complex<short> &>(inputSamples[time][2][0][0]) = std::complex<short>((short) rint(64 * cos(phi)), (short) rint(64 * sin(phi)));
	  break;

	case 2 :
          reinterpret_cast<std::complex<signed char> &>(inputSamples[time][2][0][0]) = std::complex<signed char>((signed char) rint(32 * cos(phi)), (signed char) rint(32 * sin(phi)));
	  break;
      }
    }
  }
}


static void printTestOutput(const CorrelatorParset &ps, MultiArrayHostBuffer<std::complex<int32_t>, 4> &visibilities)
{
  if (ps.nrBaselines() >= 6 /* && ps.nrOutputChannelsPerSubband() == 256 */)
#if 0
    for (int channel = 1; channel < ps.nrOutputChannelsPerSubband(); channel ++)
      std::cout << channel << ' ' << visibilities[5][channel][0] << std::endl;
#else
  {
    float max = 0.0;

    for (unsigned ch = 1; ch < ps.nrOutputChannelsPerSubband(); ch ++)
      if (abs(visibilities[5][ch][0]) > max)
	max = abs(visibilities[5][ch][0]);

    std::cout << "max = " << max << std::endl;

    for (int channel = 1; channel < ps.nrOutputChannelsPerSubband(); channel ++)
      std::cout << channel << ' ' << (10 * std::log10(abs(visibilities[5][channel][0]) / max)) << std::endl;
  }
#endif
}


static void printJgraphOutput(const CorrelatorParset &ps, MultiArrayHostBuffer<std::complex<float>, 3> &visibilities)
{
  if (ps.nrBaselines() >= 6)
  {
    std::cout << "newgraph newcurve linetype solid marktype none color 1 0 0 pts" << std::endl;

    for (int channel = 0; channel < ps.nrOutputChannelsPerSubband(); channel ++)
      std::cout << channel << ' ' << std::real(visibilities[5][channel][0]) << std::endl;
  }
}


int main(int argc, char **argv)
{
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  try {
#endif
    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

    std::cout << ">>> Running DeviceInstanceTest" << std::endl;
    CorrelatorParset	ps(argc, argv);
    CorrelatorPipeline	pipeline(ps);

    MultiArrayHostBuffer<char, 4> hostInputBuffer(boost::extents[(ps.nrSamplesPerChannel() + NR_TAPS - 1) * ps.nrChannelsPerSubbandBeforeFilter()][ps.nrStations()][ps.nrPolarizations()][ps.nrBytesPerComplexSample()]);
    MultiArrayHostBuffer<float, 3> hostDelaysAtBegin(boost::extents[ps.nrBeams()][ps.nrStations()][ps.nrPolarizations()]);
    MultiArrayHostBuffer<float, 3> hostDelaysAfterEnd(boost::extents[ps.nrBeams()][ps.nrStations()][ps.nrPolarizations()]);
    MultiArrayHostBuffer<std::complex<int32_t>, 4> hostVisibilities(boost::extents[ps.nrOutputChannelsPerSubband()][ps.nrBaselines()][ps.nrPolarizations()][ps.nrPolarizations()]);

    setTestPattern(ps, hostInputBuffer);
    memset(hostDelaysAtBegin.origin(), 0, hostDelaysAtBegin.bytesize());
    memset(hostDelaysAfterEnd.origin(), 0, hostDelaysAfterEnd.bytesize());

    pipeline.deviceInstances[0]->doSubband(TimeStamp(0, 0), 0, hostInputBuffer, hostDelaysAtBegin, hostDelaysAfterEnd, hostVisibilities);

    //printTestOutput(ps, hostVisibilities);
    printJgraphOutput(ps, hostVisibilities);
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  } catch (cu::Error &ex) {
#pragma omp critical (cerr)
    std::cerr << "caught cu::Error: " << ex.what() << std::endl;
    exit(1);
  }
#endif

  return 0;
}

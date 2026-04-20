#include "Common/Config.h"

#include "ISBI/CorrelatorPipeline.h"
#include "ISBI/Parset.h"
#include "ISBI/InputSection.h"
//#include "ISBI/SignalHandler.h"
#include "Common/Affinity.h"
#include "Common/CUDA_Support.h"


#include <list>
#include <iostream>

void printArgv(int argc, char **argv)
{
  std::clog << "command line arguments: " << argv[0];

  for (int i = 1; i < argc; i ++)
    std::clog << ' ' << argv[i];

  std::clog << std::endl;
}


void printSettings(const ISBI_Parset &ps){
  std::clog << "#receivers = " << ps.nrStations() << std::endl;
  std::clog << "#polarizations = " << ps.nrPolarizations() << std::endl;
  std::clog << "#intermediate channels/subband = " << ps.nrChannelsPerSubband() << std::endl;
  std::clog << "#output channels/subband = " << ps.nrOutputChannelsPerSubband() << std::endl;
  std::clog << "#samples/channel = " << ps.nrSamplesPerChannel() << std::endl;
  std::clog << "#bits/sample = " << ps.nrBitsPerSample() << std::endl;
  std::clog << "correlator mode = " << ps.correlationMode() << std::endl;
  std::clog << "start time = " << ps.startTime() << std::endl;
  std::clog << "intended stop time = " << ps.stopTime() << std::endl;
  std::clog << "sample rate = " << ps.sampleRate() << std::endl;
  std::clog << "subband bandwidth = " << ps.subbandBandwidth() << std::endl;
  std::clog << "max delay = " << ps.maxDelay() << std::endl;
}


int main(int argc, char **argv){

#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  try {
#endif
    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

#if defined __linux__
    setScheduler(SCHED_BATCH, 0);
#endif

    printArgv(argc, argv);
    ISBI_Parset ps(argc, argv);
    printSettings(ps);

    ISBI_CorrelatorPipeline(ps).doWork();

#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  } catch (cu::Error &error) {
    std::cerr << "caught cu::Error: " << error.what() << std::endl;
    exit(1);
  } catch (std::exception &error) {
    std::cerr << "caught std::exception: " << error.what() << std::endl;
    exit(1);
  }
#endif

  return 0;
}

#include "Common/Config.h"

#include "Common/Affinity.h"
#include "Common/CUDA_Support.h"
#include "Common/PowerSensor.h"
#include "Correlator/CorrelatorPipeline.h"
#include "Correlator/Parset.h"

#include <omp.h>

#include <iostream>

int main(int argc, char **argv)
{
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  try {
#endif
    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

#if defined __linux__
    setScheduler(SCHED_BATCH, 0);
#endif

#if 0 && defined MEASURE_POWER
    powerSensor.dump("/tmp/sensor_values");
#endif

    CorrelatorParset ps(argc, argv);
    //CorrelatorPipeline pipeline(ps).doWork();
    CorrelatorPipeline pipeline(ps);

    MultiArrayHostBuffer<char, 4> hostInputBuffer(boost::extents[ps.nrStations()][ps.nrPolarizations()][(ps.nrSamplesPerChannel() + NR_TAPS - 1) * ps.nrChannelsPerSubbandBeforeFilter()][ps.nrBytesPerRealSample()]);
    MultiArrayHostBuffer<float, 2> hostDelays(boost::extents[ps.nrStations()][2]);
    //MultiArrayHostBuffer<float, 2> hostPhaseOffsets(boost::extents[ps.nrBeams()][ps.nrPolarizations()]);
    MultiArrayHostBuffer<std::complex<int32_t>, 4> hostVisibilities(boost::extents[ps.nrOutputChannelsPerSubband()][ps.nrBaselines()][ps.nrPolarizations()][ps.nrPolarizations()]);

    double startTime = omp_get_wtime();

#pragma omp parallel num_threads(ps.nrQueuesPerGPU() * ps.nrGPUs())
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
    try {
#endif
#pragma omp barrier

      double lastTime = omp_get_wtime();
      unsigned deviceNr = omp_get_thread_num() / ps.nrQueuesPerGPU();

#if defined MEASURE_POWER
#pragma omp single
      pipeline.startState = powerSensor.read();
#endif

      for (TimeStamp currentTime = ps.startTime(); currentTime < ps.stopTime(); currentTime += ps.nrSamplesPerSubbandBeforeFilter()) {
#pragma omp single nowait
#pragma omp critical (cout)
	std::cout << "time = " << currentTime << ", late = " << (double) (TimeStamp::now(ps.clockSpeed()) - currentTime) / ps.subbandBandwidth() << "s, exec = " << omp_get_wtime() - lastTime << std::endl;
	lastTime = omp_get_wtime();

	memset(hostDelays.origin(), 0, hostDelays.bytesize());
	//memset(hostPhaseOffsets.origin(), 0, hostPhaseOffsets.bytesize());

#pragma omp for schedule(dynamic), nowait, ordered
	for (unsigned subband = 0; subband < ps.nrSubbands(); subband ++)
	  pipeline.deviceInstances[deviceNr]->doSubband(currentTime, subband, hostInputBuffer, hostDelays, hostVisibilities);
      }

#pragma omp barrier

#if defined MEASURE_POWER
#pragma omp single
      pipeline.stopState = powerSensor.read();
#endif
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
    } catch (cu::Error &error) {
#pragma omp critical (cerr)
     std::cerr << "caught cu::Error: " << error.what() << std::endl;
      exit(1);
    } catch (std::exception &error) {
#pragma omp critical (cerr)
      std::cerr << "caught std::exception: " << error.what() << std::endl;
      exit(1);
    }
#endif

    double runTime = omp_get_wtime() - startTime;

#pragma omp critical (cout)
    {
      std::cout << "total: " << runTime << " s"
#if defined MEASURE_POWER
		", " << PowerSensor3::Joules(pipeline.startState, pipeline.stopState) << " J"
#endif
	        << std::endl;
    }

#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  } catch (cu::Error &ex) {
    std::cerr << "caught cu::Error: " << ex.what() << std::endl;
    exit(1);
  } catch (std::exception &error) {
    std::cerr << "caught std::exception: " << error.what() << std::endl;
    exit(1);
  }
#endif

  return 0;
}

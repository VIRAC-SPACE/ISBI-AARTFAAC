#include "Common/Config.h"

#include "ISBI/CorrelatorPipeline.h"
#include "ISBI/CorrelatorWorkQueue.h"
#include "Common/Exceptions/Exception.h"
#include "Common/CUDA_Support.h"

#include <algorithm>
#include <iostream>
#include <omp.h>


volatile std::sig_atomic_t ISBI_CorrelatorPipeline::signalCaught = false;


ISBI_CorrelatorPipeline::ISBI_CorrelatorPipeline(const ISBI_Parset &ps)
:
  CorrelatorPipeline(ps),
  ps(ps),
  nrWorkQueues(ps.nrQueuesPerGPU() * ps.nrGPUs()),
  delayCorrection(ps),
  inputSection(ps),
  outputSection(ps),
  nextTime(ps.startTime())
{
}


void ISBI_CorrelatorPipeline::doWork()
{
  double startTime = omp_get_wtime();

#pragma omp parallel num_threads(ps.nrQueuesPerGPU() * ps.nrGPUs())
  {
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
    try {
#endif
      unsigned deviceNr = omp_get_thread_num() / ps.nrQueuesPerGPU();
      deviceInstances[deviceNr]->context.setCurrent();
      CorrelatorWorkQueue(*this, *deviceInstances[deviceNr]).doWork();
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
  }

  double runTime = omp_get_wtime() - startTime;

#pragma omp critical (cout)
  std::cout << "total: " << runTime << " s" << std::endl;
}


bool ISBI_CorrelatorPipeline::getWork(unsigned preferredNode, TimeStamp &time, unsigned &subband)
{
  if (signalCaught)
    return false;

  std::lock_guard<std::mutex> lock(getWorkLock);

  // try to find a subband that belongs to this NUMA domain
  if (ps.outputBufferNodes().size() > 0)
    for (subband = 0; subband < ps.nrSubbands(); subband ++)
      if (!subbandsDone[subband] && ps.outputBufferNodes()[subband] == preferredNode)
	goto found;

  // not found ==> find a subband from another NUMA domain
  for (subband = 0; subband < ps.nrSubbands(); subband ++)
    if (!subbandsDone[subband])
      break;

#pragma omp critical (clog)
  std::clog << nextTime << ": subband " << subband << " processed on unpreferred NUMA domain " << preferredNode << std::endl;

found:
  time = nextTime;
  subbandsDone[subband] = true;

  if (subbandsDone.count() == ps.nrSubbands()) {
    nextTime += ps.nrSamplesPerSubbandBeforeFilter();
    subbandsDone.reset();
  }

  return time < ps.stopTime();
}


void ISBI_CorrelatorPipeline::logProgress(const TimeStamp &time) const
{
  static double lastTime = omp_get_wtime();

#pragma omp critical (clog)
  {
    std::clog << "time: " << time << ", ";

    if (ps.realTime())
      std::clog << "late: " << (double) TimeStamp::now(ps.clockSpeed()) - (double) time - ps.nrSamplesPerSubbandBeforeFilter() / ps.subbandBandwidth() << "s, ";

    std::clog << "exec: " << omp_get_wtime() - lastTime << std::endl;
  }

  lastTime = omp_get_wtime();
}


void ISBI_CorrelatorPipeline::startReadTransaction(const TimeStamp &time)
{
  // This is not the most elegant way to find out find out which thread is the first/last
  // to process TimeStamp, but it works
  // currentTimes keeps track of which times are in flight

  bool iAmTheFirstOne = false;

  {
    std::lock_guard<std::mutex> lock(currentTimesMutex);

    if (std::find(currentTimes.begin(), currentTimes.end(), time) == currentTimes.end()) {
      // I am the first thread that processes this TimeStamp
      iAmTheFirstOne = true;
      currentTimes.insert(currentTimes.end(), ps.nrSubbands(), time);
    }
  }

  if (iAmTheFirstOne) {
    inputSection.startReadTransaction(time);
    currentTime.advanceTo(time);
    logProgress(time);
  } else {
    currentTime.waitFor(time);
  }
}


void ISBI_CorrelatorPipeline::endReadTransaction(const TimeStamp &time)
{
  std::lock_guard<std::mutex> lock(currentTimesMutex);

  currentTimes.erase(std::find(currentTimes.begin(), currentTimes.end(), time));

  if (std::find(currentTimes.begin(), currentTimes.end(), time) == currentTimes.end()) // I am the last thread that processes this TimeStamp
    inputSection.endReadTransaction(time);
}


void ISBI_CorrelatorPipeline::caughtSignal()
{
  signalCaught = true;
  InputBuffer::caughtSignal();
}

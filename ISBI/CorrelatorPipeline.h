#ifndef ISBI_CORRELATOR_PIPELINE
#define ISBI_CORRELATOR_PIPELINE

#include "ISBI/InputSection.h"
#include "ISBI/OutputSection.h"
#include "ISBI/DelayCorrection.h"

#include <libfilter/FilterBank.h>
#include "Common/PerformanceCounter.h"
#include "Common/SlidingPointer.h"
#include "Correlator/CorrelatorPipeline.h"

#include <bitset>
#include <csignal>
#include <string>
#include <mutex>
#include <vector>


class ISBI_CorrelatorPipeline : public CorrelatorPipeline
{
  public:
			   ISBI_CorrelatorPipeline(const ISBI_Parset &);

    bool		   getWork(unsigned preferredNode, TimeStamp &, unsigned &subband);
    void		   doWork();

    void		   startReadTransaction(const TimeStamp &);
    void		   endReadTransaction(const TimeStamp &);

    static void		   caughtSignal();

    const ISBI_Parset  &ps;
    const unsigned	   nrWorkQueues;
    DelayCorrection delayCorrection;
    InputSection	   inputSection;
    OutputSection	   outputSection;

    std::vector<TimeStamp> currentTimes;
    std::mutex		   currentTimesMutex;
    SlidingPointer<TimeStamp> currentTime;

  private:
    void		   logProgress(const TimeStamp &time) const;

    std::mutex		   getWorkLock;
    std::bitset<64>	   subbandsDone;
    TimeStamp		   nextTime;

    static volatile std::sig_atomic_t signalCaught;
};

#endif

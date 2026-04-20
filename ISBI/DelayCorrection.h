#ifndef ISBI_DELAY_CORRECTION_H
#define ISBI_DELAY_CORRECTION_H

#include "ISBI/Parset.h"
#include "Common/TimeStamp.h"

#include <vector>
#include <cstdint>
#include <map>

class DelayCorrection {
  public:
    struct StationDelay {
      int64_t integerSamples = 0;
      float d0 = 0.0f;
      float d1 = 0.0f;
    };

    DelayCorrection(const ISBI_Parset &);

    std::vector<StationDelay> stationDelays(const TimeStamp &) const;

  private:
    const ISBI_Parset &ps;
    unsigned referenceStation;

    std::vector<std::map<int64_t, double>> rawDelays;

    std::vector<std::map<int64_t, double>> readDelayFiles() const;

    double getDelayAt(const TimeStamp &, unsigned station) const;
};

#endif

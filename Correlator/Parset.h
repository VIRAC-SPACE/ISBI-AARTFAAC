#ifndef CORRELATOR_PARSET_H
#define CORRELATOR_PARSET_H

#include "Common/Parset.h"


class CorrelatorParset : public Parset
{
  public:
    CorrelatorParset(int argc, char **argv, bool throwExceptionOnUnmatchedParameter = true);

    unsigned nrVisibilityPolarizations() const { return _nrVisibilityPolarizations; }
    unsigned correlationMode() const { return _correlationMode; }
    unsigned nrBaselines() const { return nrStations() * (nrStations() + 1) / 2; }
    unsigned nrOutputChannelsPerSubband() const { return _nrOutputChannelsPerSubband; }
    unsigned channelIntegrationFactor() const { return nrChannelsPerSubband() == 1 ? 1 : (nrChannelsPerSubband() - 1) / nrOutputChannelsPerSubband(); }
    unsigned outputChannelBandwidth() const { return channelBandwidth() * channelIntegrationFactor(); }

    virtual std::vector<std::string> compileOptions() const;

  protected:
    unsigned _correlationMode;
    unsigned _nrVisibilityPolarizations;
    unsigned _nrOutputChannelsPerSubband;

};

#endif

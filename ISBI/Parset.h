#if !defined ISBI_PARSET_H
#define ISBI_PARSET_H

#include "Correlator/Parset.h"

class ISBI_Parset : public CorrelatorParset
{
  public:
    ISBI_Parset(int argc, char **argv);

    const std::vector<std::string> &inputDescriptors() const { return _inputDescriptors; }
    const std::vector<std::string> &outputDescriptors() const { return _outputDescriptors; }

#if defined __linux__
    std::vector<unsigned>  inputBufferNodes() const { return _inputBufferNodes; }
    std::vector<unsigned>  outputBufferNodes() const { return _outputBufferNodes; }
#endif

    unsigned visibilitiesIntegration() const { return _visibilitiesIntegration; }
    unsigned nrRingBufferSamplesPerSubband() const { return _nrRingBufferSamplesPerSubband; }

    const int maxDelay() const { return _maxDelaySamples; }; 
    const std::string delayFile() const { return _delayFile; }
    
    virtual std::vector<std::string> compileOptions() const;
      std::string _delayFile;

  private:
    std::vector<std::string> _inputDescriptors, _outputDescriptors;

#if defined __linux__
    std::vector<unsigned> _inputBufferNodes, _outputBufferNodes;
#endif

    unsigned _nrRingBufferSamplesPerSubband;
    unsigned _visibilitiesIntegration;
    int _maxDelaySamples;
};


#endif

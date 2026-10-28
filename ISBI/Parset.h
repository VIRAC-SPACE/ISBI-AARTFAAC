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
    // Sample-times carried by one VDIF frame, per station. 0 means "take it
    // from each frame's own header", which is the default and is what makes a
    // mixed-mode array work: VDIFHeader::samplesPerFrame() is
    // dataSize * 8 / (bitsPerSample * nrChannels), so an 8-channel 2-bit 8032 B
    // frame gives 4000 and a 1-channel one gives 32000. Only override it when a
    // recorder writes a header that disagrees with its own payload.
    unsigned samplesPerFrame(unsigned station) const
      { return _samplesPerFrames.empty() ? 0 :
               _samplesPerFrames[station < _samplesPerFrames.size() ? station : 0]; };

    // Where one (subband, polarisation) of one station comes from in that
    // station's recording: a VDIF thread and a channel within that thread's
    // frames. Single-thread recordings (E011, and Ef/Hh/Ir in n26l2) use
    // thread 0 throughout; Cm and Da in n26l2 carry one channel per thread, so
    // the thread is the only thing distinguishing their two polarisations.
    struct ChannelSelector
    {
      unsigned thread;
      unsigned channel;
    };

    // Selector for each (subband, polarisation), in that order, for one
    // station. Backends differ: in n26l2, Ir has its two circular polarisations
    // exchanged relative to Ef and Hh (its channel n carries their channel n^2),
    // so this has to be settable per station, not once for the array.
    const std::vector<ChannelSelector> &channelMapping(unsigned station) const
      { return _channelMappings[station < _channelMappings.size() ? station : 0]; };

    // Station that every produced CROSS baseline must include, or -1 for all
    // baselines. Cross baselines not involving it are zeroed and given zero
    // weight: across a mixed-mode array only the reference's grid is common to
    // everyone, so any other pair is not a meaningful correlation.
    // Autocorrelations are always kept, whatever the reference is -- they
    // belong to one station rather than to a pair, they cost nothing extra, and
    // normalising a kept baseline to a coherence needs the autocorrelation of
    // BOTH its stations, so dropping them would break the baselines we keep.
    int referenceStation() const { return _referenceStation; };
    bool producesBaseline(unsigned stat1, unsigned stat2) const
      { return _referenceStation < 0 || stat1 == stat2 ||
               stat1 == (unsigned) _referenceStation || stat2 == (unsigned) _referenceStation; };

    // True sample rate of each station's recording, which the VDIF header does
    // not carry. Defaults to the global sampleRate(), i.e. no decimation.
    unsigned stationSampleRate(unsigned station) const
      { return _stationSampleRates.empty() ? sampleRate() :
               _stationSampleRates[station < _stationSampleRates.size() ? station : 0]; };

    // Keep 1 input sample in every decimation(). A station recorded faster than
    // the correlator's global rate is brought down to it by plain decimation,
    // with NO anti-alias filter -- so the half of its band that is thrown away
    // folds back on top of the half that is kept, reversed about the new
    // Nyquist. That is deliberate: it costs sqrt(decimation) in coherence
    // because the folded copy is uncorrelated noise on a mixed baseline, and in
    // exchange the station needs no resampling, no second filter configuration
    // and no separate time base -- after decimation it simply IS a station at
    // the global rate. The folded copy is still usable: point a second subband
    // at the same channel and it correlates against the mirrored sky band.
    unsigned decimation(unsigned station) const
      { return stationSampleRate(station) / sampleRate(); };

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
    std::vector<unsigned> _samplesPerFrames;
    std::vector<unsigned> _stationSampleRates;
    std::vector<std::vector<ChannelSelector>> _channelMappings;
    int _referenceStation;
};


#endif

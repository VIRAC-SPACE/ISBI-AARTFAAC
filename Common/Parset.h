#ifndef PARSET_H
#define PARSET_H

#include "Common/TimeStamp.h"

#include <exception>
#include <string>
#include <vector>
#include <map>

#include <boost/lexical_cast.hpp>

#if defined __linux__
#include <sched.h>
#endif


class Parset
{
  public:
    Parset(int argc, char **argv, bool throwExceptionOnUnmatchedParameter = true);

    class Error : public std::exception {
      public:
        Error(const std::string &msg) : msg(msg) {}
	virtual ~Error() noexcept {}

	virtual const char *what() const noexcept { return msg.c_str(); }

      private:
        std::string msg;
    };

    unsigned nrGPUs() const { return _GPUs.size(); }
    const std::vector<unsigned> &GPUs() const { return _GPUs; }
    unsigned nrQueuesPerGPU() const { return _nrQueuesPerGPU; }
    unsigned nrPolarizations() const { return _nrPolarizations; }
    unsigned nrStations() const { return _nrStations; }
    unsigned clockSpeed() const { return _clockSpeed; }
    unsigned nrSubbands() const { return _nrSubbands; }
    const std::vector<double> &subbandFrequencies() const { return _subbandFrequencies; }
    double   subbandBandwidth() const { return _subbandBandwidth; }
    double   channelBandwidth() const { return subbandBandwidth() / nrChannelsPerSubband(); }
    unsigned   sampleRate() const { return _sampleRate; }
    unsigned nrChannelsPerSubband() const { return _nrChannelsPerSubband; } 
    unsigned nrChannelsPerSubbandBeforeFilter() const { return 2 * _nrChannelsPerSubband; } 
    unsigned nrBitsPerSample() const { return _nrBitsPerSample; }
    unsigned nrBytesPerRealSample() const { return 1; }
    unsigned nrBytesPerComplexSample() const { return 2 * nrBitsPerSample() / 8; }
    unsigned nrBeams() const { return _nrBeams; }
    unsigned nrSamplesPerChannel() const { return _nrSamplesPerChannel; }
    unsigned nrSamplesPerSubband() const { return nrSamplesPerChannel() * nrChannelsPerSubband(); }
    unsigned nrSamplesPerSubbandBeforeFilter() const { return nrSamplesPerChannel() * nrChannelsPerSubbandBeforeFilter(); }
    TimeStamp startTime() const { return _startTime; };
    TimeStamp stopTime() const { return _stopTime; };
    bool     correctBandPass() const { return _correctBandPass; }
    bool     delayCompensation() const { return _delayCompensation; }
    bool     profiling() const { return _profiling; }

    bool     realTime() const { return _realTime; }

#if defined __linux__
    cpu_set_t allowedCPUs() const;
    cpu_set_t allowedCPUs(unsigned node) const { return _allowedCPUs[node]; }
#endif

    virtual std::vector<std::string> compileOptions() const;

  protected:
    template <typename T> static std::vector<T> splitArgs(const std::string &args, char delimiter = ',')
    {
      std::vector<T> result;
      std::stringstream iss(args);
      std::string token;

      while (std::getline(iss, token, delimiter))
	result.push_back(boost::lexical_cast<T>(token));

      return result;
    }

    void setGPUs(const char *arg);
    void handleSubbandsAndFrequencies();

#if defined __linux__
    static cpu_set_t cpu_set(const char *start, const char *stop);
    static std::vector<cpu_set_t> getAffinityVector(const char *arg);
    static std::vector<unsigned> getNodeVector(const char *arg);
#endif

    std::vector<unsigned> _GPUs;
    unsigned _nrQueuesPerGPU;
    unsigned _nrStations;
    unsigned _nrSubbands;
    std::vector<double> _subbandFrequencies;
    std::vector<unsigned> _subbandNumbers;
    unsigned _nrPolarizations;
    unsigned _clockSpeed;
    double   _subbandBandwidth;
    unsigned   _sampleRate;
    unsigned _nrChannelsPerSubband;
    unsigned _nrBitsPerSample;
    unsigned _nrBeams;
    unsigned _nrSamplesPerChannel;
    bool     _realTime;
    bool     _correctBandPass;
    bool     _delayCompensation;
    bool     _profiling;
    TimeStamp _startTime, _stopTime;

#if defined __linux__
    std::vector<cpu_set_t> _allowedCPUs;
#endif

  protected:
    std::vector<std::string> toPassFurther;
};

#endif

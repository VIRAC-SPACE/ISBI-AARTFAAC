#include "Common/Config.h"
#include "Common/Parset.h"

#include <boost/program_options.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>


void Parset::setGPUs(const char *arg)
{
  _GPUs.clear();

  do {
    unsigned begin, end;

    switch (sscanf(arg, "%u-%u", &begin, &end)) {
      case 0 :	throw Error("cannot parse GPU list");

      case 1 :	_GPUs.push_back(begin);
		break;

      case 2 :	for (unsigned gpu = begin; gpu <= end; gpu ++)
		  _GPUs.push_back(gpu);

		break;
    }
  } while ((arg = strchr(arg, ',') + 1) != (char *) 1);
}


#if defined __linux__

cpu_set_t Parset::cpu_set(const char *start, const char *stop)
{
  cpu_set_t set;
  CPU_ZERO(&set);

  do {
    unsigned begin, end;

    switch (sscanf(start, "%u-%u", &begin, &end)) {
      case 0 :	throw Error("cannot parse CPU list");

      case 1 :	CPU_SET(begin, &set);
		break;

      case 2 :	for (unsigned cpu = begin; cpu <= end; cpu ++)
		  CPU_SET(cpu, &set);

		break;
    }
  } while ((start = strchr(start, ',') + 1) != (char *) 1 && start < stop);

  return set;
}


std::vector<cpu_set_t> Parset::getAffinityVector(const char *arg)
{
  std::vector<cpu_set_t> affinity;
  const char *start = arg;

  do {
    if (*arg == '/' || *arg == '\0') {
      affinity.push_back(cpu_set(start, arg));
      start = arg + 1;
    }
  } while (*arg ++ != '\0');

  return affinity;
}


std::vector<unsigned> Parset::getNodeVector(const char *arg)
{
  std::vector<unsigned> nodes;

  do {
    unsigned node, repetitions;

    switch (sscanf(arg, "%u:%u", &node, &repetitions)) {
      case 0 :	throw Error("cannot parse node list");

      case 1 :	nodes.push_back(node);
		break;

      case 2 :	nodes.insert(nodes.end(), repetitions, node);
		break;
    }
  } while ((arg = strchr(arg, ',') + 1) != (char *) 1);

  return nodes;
}


cpu_set_t Parset::allowedCPUs() const
{
  cpu_set_t cpus;
  CPU_ZERO(&cpus);

  for (cpu_set_t c : _allowedCPUs)
    CPU_OR(&cpus, &cpus, &c);

  return cpus;
}

#endif


void Parset::handleSubbandsAndFrequencies()
{

  bool hasSubbandFrequencies = _subbandFrequencies.size() > 0, hasSubbandNumbers = _subbandNumbers.size() > 0;

  if (hasSubbandFrequencies && hasSubbandNumbers)
    throw Error("should not provide both subband frequencies and subband numbers");

  if (!hasSubbandFrequencies)
    _subbandFrequencies.resize(_nrSubbands, 0);
  else if (_subbandFrequencies.size() != _nrSubbands)
    throw Error(std::string("expected ") + boost::lexical_cast<std::string>(_nrSubbands) + " subband frequencies, but got " + boost::lexical_cast<std::string>(_subbandFrequencies.size()));

  if (!hasSubbandNumbers)
    _subbandNumbers.resize(_nrSubbands, 0);
  else if (_subbandNumbers.size() != _nrSubbands)
    throw Error(std::string("expected ") + boost::lexical_cast<std::string>(_nrSubbands) + " subband numbers, but got " + boost::lexical_cast<std::string>(_subbandNumbers.size()));

  if (hasSubbandNumbers)
    for (unsigned subband = 0; subband < _nrSubbands; subband ++)
      _subbandFrequencies[subband] = _subbandNumbers[subband] * _subbandBandwidth;

  if (hasSubbandFrequencies)
    for (unsigned subband = 0; subband < _nrSubbands; subband ++)
      _subbandNumbers[subband] = rint(_subbandFrequencies[subband] / _subbandBandwidth);
}


Parset::Parset(int argc, char **argv, bool throwExceptionOnUnmatchedParameter)
:
  _sampleRate(_clockSpeed),
  _nrBeams(1)
{
  _GPUs.push_back(0);

#if defined __linux__
  for (unsigned node = 0;; node ++) {
    char path[64];
    sprintf(path, "/sys/devices/system/node/node%u/cpulist", node);
    std::ifstream stream(path);

    if (!stream.good())
      break;

    std::string cpulist;
    stream >> cpulist;
    _allowedCPUs.push_back(cpu_set(cpulist.c_str(), cpulist.c_str() + cpulist.size()));
  }
#endif

  using namespace boost::program_options;

  options_description allowed_options;
  unsigned runTime;
  std::string providedStartTime;

  allowed_options.add_options()
    ("nrBitsPerSample,b", value<unsigned>(&_nrBitsPerSample)->default_value(8))
    ("correctBandPass,B", value<bool>(&_correctBandPass)->default_value(true))
    ("nrChannelsPerSubband,c", value<unsigned>(&_nrChannelsPerSubband)->default_value(64))
    ("delayCompensation,d", value<bool>(&_delayCompensation)->default_value(false))
    ("startTime,D", value<std::string>()->notifier([&providedStartTime] (std::string arg) { /* work-around for CodeXL that cannot pass white space */ std::replace(arg.begin(), arg.end(), '_', ' '); providedStartTime = arg; } ))
    ("clockSpeed,f", value<unsigned>(&_clockSpeed)->default_value(32000000))
    ("subbandBandwidth,fs", value<double>(&_subbandBandwidth)->default_value(8e6))
    ("subbandFrequencies,F", value<std::string>()->notifier([this] (const std::string &arg) { _subbandFrequencies = splitArgs<double>(arg); } ))
    ("GPUs,g", value<std::string>()->notifier([this] (const std::string &arg) { setGPUs(arg.c_str()); } ))
    ("nrStations,n", value<unsigned>(&_nrStations)->default_value(288))
#if defined __linux__
    ("allowedCPUs,N", value<std::string>()->notifier([this] (const std::string &arg) { _allowedCPUs = getAffinityVector(arg.c_str()); } ))
#endif
    ("profiling,p", value<bool>(&_profiling)->default_value(false))
    ("nrPolarizations", value<unsigned>(&_nrPolarizations)->default_value(2))
    ("nrQueuesPerGPU,q", value<unsigned>(&_nrQueuesPerGPU)->default_value(2))
    ("runTime,r", value<unsigned>(&runTime)->default_value(20))
    ("realTime,R", value<bool>(&_realTime)->default_value(false)) // real-time mode reads VDIF frames from UDP, otherwise from files
    ("nrSubbands,s", value<unsigned>(&_nrSubbands)->default_value(16))
    ("subbandNumbers,S", value<std::string>()->notifier([this] (const std::string &arg) { _subbandNumbers = splitArgs<unsigned>(arg); } ))
    ("nrSamplesPerChannel,t", value<unsigned>(&_nrSamplesPerChannel)->default_value(3072))
  ;

  variables_map vm;
  parsed_options parsed = command_line_parser(argc, argv).options(allowed_options).allow_unregistered().run();
  toPassFurther = collect_unrecognized(parsed.options, include_positional);
  store(parsed, vm);
  notify(vm);

  if (throwExceptionOnUnmatchedParameter && toPassFurther.size() > 0)
    throw Error(std::string("unrecognized argument \'") + toPassFurther[0] + '\'');

  handleSubbandsAndFrequencies();
  _sampleRate = static_cast<double>(clockSpeed()); 
  _startTime = providedStartTime != "" ? TimeStamp::fromDate(providedStartTime.c_str(), clockSpeed()) : TimeStamp::now(clockSpeed()) + 30 * clockSpeed();
  _stopTime  = _startTime + static_cast<int64_t>(runTime) * clockSpeed();
}


std::vector<std::string> Parset::compileOptions() const
{
  std::vector<std::string> options =
  {
    "-DNR_BITS=" + std::to_string(nrBitsPerSample()),
    "-DNR_CHANNELS_PER_SUBBAND=" + std::to_string(nrChannelsPerSubband()),
    "-DNR_POLARIZATIONS=" + std::to_string(nrPolarizations()),
    "-DNR_RECEIVERS=" + std::to_string(nrStations()),
    "-DNR_SAMPLES_PER_CHANNEL=" + std::to_string(nrSamplesPerChannel()),
    "-DNR_TAPS=" + std::to_string(NR_TAPS),
    "-DSUBBAND_BANDWIDTH=" + std::to_string(subbandBandwidth()),
    //"-DNR_RING_BUFFER_SAMPLES_PER_SUBBAND=" + std::to_string(nrRingBufferSamplesPerSubband()),
  };

  return options;
}
#include "ISBI/Parset.h"
#include <boost/program_options.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>


#if 0
std::vector<std::string> AARTFAAC_Parset::getDescriptors(const std::string &arg)
{
  // TODO: commas in filenames are not supported

  std::vector<std::string> descriptors;
  std::istringstream iss(arg);
  std::string token;

  while (std::getline(iss, token, ','))
    descriptors.push_back(token);

  return descriptors;
}
#endif


ISBI_Parset::ISBI_Parset(int argc, char **argv)
:
  CorrelatorParset(argc, argv, false),
  _nrRingBufferSamplesPerSubband(128015360),
  _visibilitiesIntegration(1),
  _maxDelaySamples(1000),
  // Empty means "ask each frame's header", which reproduces the former
  // hard-coded 2000 for E011's 16-channel 2-bit frames without being told.
  _samplesPerFrames(),
  _stationSampleRates(),
  // Default is the E011 16-channel order on thread 0, so existing invocations
  // are unchanged.
  _channelMappings({{{0,  8}, {0, 12}, {0, 0}, {0, 4}, {0,  9}, {0, 13}, {0, 1}, {0, 5},
                     {0, 10}, {0, 14}, {0, 2}, {0, 6}, {0, 11}, {0, 15}, {0, 3}, {0, 7}}}),
  _referenceStation(-1)
{
  using namespace boost::program_options;

  options_description allowed_options;

  allowed_options.add_options()
    ("inputDescriptors,i", value<std::string>()->notifier([this] (std::string arg) { _inputDescriptors = splitArgs<std::string>(arg); } ))
    ("outputDescriptors,o", value<std::string>()->notifier([this] (std::string arg) { _outputDescriptors = splitArgs<std::string>(arg); } ))
#if defined __linux__
    ("inputBufferNodes,A", value<std::string>()->notifier([this] (std::string arg) { _inputBufferNodes = getNodeVector(arg.c_str()); }))
    ("outputBufferNodes,O", value<std::string>()->notifier([this] (std::string arg) { _outputBufferNodes = getNodeVector(arg.c_str()); }))
#endif
    ("nrRingBufferSamplesPerSubband,T", value<unsigned>(&_nrRingBufferSamplesPerSubband))
    ("visibilitiesIntegration,I", value<unsigned>(&_visibilitiesIntegration))
    ("delayFile,delayFile", value<std::string>()->notifier([this] (std::string arg) { _delayFile = arg; } ))
    ("maxDelay,maxDelay", value<int>(&_maxDelaySamples))
    ("samplesPerFrame,samplesPerFrame", value<std::string>()->notifier([this] (const std::string &arg) {
        // One value applies to every station; "4000;32000" gives station 0 the
        // first and station 1 the second. Only needed when a header lies.
        _samplesPerFrames = splitArgs<unsigned>(arg, ';');
        if (_samplesPerFrames.empty())
          throw Error("--samplesPerFrame is empty");
      } ))
    ("referenceStation,referenceStation", value<int>(&_referenceStation))
    ("stationSampleRate,stationSampleRate", value<std::string>()->notifier([this] (const std::string &arg) {
        // ';'-separated, one per station. A station faster than -f is decimated
        // down to it on input. Must be an integer multiple of -f.
        _stationSampleRates = splitArgs<unsigned>(arg, ';');
        if (_stationSampleRates.empty())
          throw Error("--stationSampleRate is empty");
      } ))
    ("channelMapping,channelMapping", value<std::string>()->notifier([this] (const std::string &arg) {
        // "0,1,2,3" applies to every station; "0,1;2,3" gives station 0 the
        // first list and station 1 the second, and so on. An entry is a channel
        // within the frame, or "thread.channel" when the station splits its
        // channels over several VDIF threads instead of interleaving them in
        // one frame (n26l2's Cm and Da are "0.0,1.0").
        _channelMappings.clear();
        std::string group;
        std::istringstream groups(arg);

        while (std::getline(groups, group, ';')) {
          if (group.empty())
            continue;

          std::vector<ChannelSelector> selectors;

          for (const std::string &entry : splitArgs<std::string>(group)) {
            std::string::size_type dot = entry.find('.');
            ChannelSelector selector{0, 0};

            try {
              if (dot == std::string::npos) {
                selector.channel = std::stoul(entry);
              } else {
                selector.thread  = std::stoul(entry.substr(0, dot));
                selector.channel = std::stoul(entry.substr(dot + 1));
              }
            } catch (const std::exception &) {
              throw Error("--channelMapping entry '" + entry + "' is not a channel or thread.channel");
            }

            selectors.push_back(selector);
          }

          _channelMappings.push_back(selectors);
        }

        if (_channelMappings.empty())
          throw Error("--channelMapping is empty");
      } ))
  ;


  variables_map vm;
  parsed_options parsed = command_line_parser(toPassFurther).options(allowed_options).allow_unregistered().run();
  toPassFurther = collect_unrecognized(parsed.options, include_positional);
  store(parsed, vm);
  notify(vm);

  if (toPassFurther.size() > 0)
    throw Error(std::string("unrecognized argument \'") + toPassFurther[0] + '\'');

  // One input descriptor per station, exactly. Too few leaves stations with no
  // data; too many makes InputSection build an InputBuffer whose station index
  // is past the end of the ring buffers, which used to segfault with no
  // explanation rather than say what was wrong.
  if (_inputDescriptors.size() != nrStations())
    throw Error("-n says " + std::to_string(nrStations()) + " stations but -i gives " +
                std::to_string(_inputDescriptors.size()) + " input descriptor" +
                (_inputDescriptors.size() == 1 ? "" : "s"));

  for (unsigned station = 0; station < nrStations(); station ++) {
    unsigned rate = stationSampleRate(station);

    if (rate < sampleRate() || rate % sampleRate() != 0)
      throw Error("--stationSampleRate for station " + std::to_string(station) + " is " +
                  std::to_string(rate) + ", which is not an integer multiple of the correlator's " +
                  std::to_string(sampleRate()) + " (-f). Set -f to the LOWEST station rate.");
  }

  if (_referenceStation >= (int) nrStations())
    throw Error("--referenceStation " + std::to_string(_referenceStation) +
                " is not one of the " + std::to_string(nrStations()) + " stations");


#if defined __linux__
  if (_inputBufferNodes.size() != 0 && _inputBufferNodes.size() != _inputDescriptors.size())
    throw Error("input buffer node list has unexpected size");

  if (_outputBufferNodes.size() != 0 && _outputBufferNodes.size() != _outputDescriptors.size())
    throw Error("output buffer node list has unexpected size");
#endif

}


std::vector<std::string> ISBI_Parset::compileOptions() const
{
  std::vector<std::string> options =
  {
    "-DNR_RING_BUFFER_SAMPLES_PER_SUBBAND=" + std::to_string(nrRingBufferSamplesPerSubband()),
  };

  std::vector<std::string> parentOptions = CorrelatorParset::compileOptions();
  options.insert(options.end(), parentOptions.begin(), parentOptions.end());
  return options;
}

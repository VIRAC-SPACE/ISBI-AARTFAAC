#include "ISBI/Parset.h"
#include <boost/program_options.hpp>
#include <fstream>
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
  _maxDelaySamples(1000)
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
  ;


  variables_map vm;
  parsed_options parsed = command_line_parser(toPassFurther).options(allowed_options).allow_unregistered().run();
  toPassFurther = collect_unrecognized(parsed.options, include_positional);
  store(parsed, vm);
  notify(vm);

  if (toPassFurther.size() > 0)
    throw Error(std::string("unrecognized argument \'") + toPassFurther[0] + '\'');


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

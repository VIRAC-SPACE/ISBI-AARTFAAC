#include "ISBI/Parset.h"

#include <boost/program_options.hpp>
#include <fstream>


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
  _nrRingBufferSamplesPerSubband(2 * 16000000 + 256),
  _visibilitiesIntegration(1)
{
  using namespace boost::program_options;

  // std::string delayPath;

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
    ("channelMapping,M", value<std::vector<int>>(&_channelMapping)->multitoken())
    // ("delayPath,K", value<std::string>()->notifier([&delayPath] (const std::string &arg) { delayPath = arg; } ))
  ;


  variables_map vm;
  parsed_options parsed = command_line_parser(toPassFurther).options(allowed_options).allow_unregistered().run();
  toPassFurther = collect_unrecognized(parsed.options, include_positional);
  store(parsed, vm);
  notify(vm);

  // std::ifstream delayFile(delayPath, std::ios::binary);

  // uint32_t num_rows, num_cols;
  // delayFile.read(reinterpret_cast<char*>(&num_rows), sizeof(uint32_t));
  // delayFile.read(reinterpret_cast<char*>(&num_cols), sizeof(uint32_t));
  // 
  // _trueDelays = std::vector<int>(num_rows * num_cols);
  // _fracDelays = std::vector<double>(num_rows * num_cols);

  // for (uint32_t i = 0; i < num_rows; i++)
  //   delayFile.read(reinterpret_cast<char*>(_trueDelays[i * num_cols].data()), num_cols * sizeof(int));

  // for (uint32_t i = 0; i < num_rows; i++)
  //   delayFile.read(reinterpret_cast<char*>(_fracDelays[i * num_cols].data()), num_cols * sizeof(double));


  // uint32_t num_frequencies;
  // delayFile.read(reinterpret_cast<char*>(&num_frequencies), sizeof(uint32_t));

  // _centerFrequencies = std::vector<double>(num_frequencies, 0);

  // delayFile.read(reinterpret_cast<char*>(_centerFrequencies.data()), num_frequencies * sizeof(double));

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

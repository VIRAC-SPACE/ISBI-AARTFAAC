#include "Correlator/Parset.h"

#include <boost/program_options.hpp>

#include <fstream>

CorrelatorParset::CorrelatorParset(int argc, char **argv, bool throwExceptionOnUnmatchedParameter)
:
  Parset(argc, argv, false)
{
  using namespace boost::program_options;

  options_description allowed_options;

  allowed_options.add_options()
    ("nrOutputChannelsPerSubband,C", value<unsigned>(&_nrOutputChannelsPerSubband)->default_value(0))
    ("correlationMode,m", value<unsigned>(&_correlationMode)->default_value(0xF))
    ("delayFile,delayFile", value<std::string>()->notifier([this] (std::string arg) { delayFile = arg; } ))
  ;

  variables_map vm;
  parsed_options parsed = command_line_parser(toPassFurther).options(allowed_options).allow_unregistered().run();
  toPassFurther = collect_unrecognized(parsed.options, include_positional);
  store(parsed, vm);
  notify(vm);

  if (throwExceptionOnUnmatchedParameter && toPassFurther.size() > 0)
    throw Error(std::string("unrecognized argument \'") + toPassFurther[0] + '\'');


  if (_nrOutputChannelsPerSubband == 0)
    _nrOutputChannelsPerSubband = std::max(_nrChannelsPerSubband - 1, 1U);

  if (_nrOutputChannelsPerSubband > _nrChannelsPerSubband)
    throw Error("#output channels cannot exceed #internal channels");

  switch (_nrPolarizations) {
    case 1: _nrVisibilityPolarizations = 1;
	    break;

    case 2: switch (_correlationMode) {
	      case 0x1:
	      case 0x8: _nrVisibilityPolarizations = 1;
			break;

	      case 0x9: _nrVisibilityPolarizations = 2;
			break;

	      case 0xF: _nrVisibilityPolarizations = 4;
			break;

	      default:  throw Error("unsupported correlation mode");
	    }

	    break;

    default: throw Error("unsupported #polarizations");
  }
}


std::vector<std::string> CorrelatorParset::compileOptions() const
{
  std::vector<std::string> options =
  {
    "-DNR_OUTPUT_CHANNELS_PER_SUBBAND=" + std::to_string(nrOutputChannelsPerSubband()),
  };

  std::vector<std::string> parentOptions = Parset::compileOptions();
  options.insert(options.end(), parentOptions.begin(), parentOptions.end());
  return options;
}

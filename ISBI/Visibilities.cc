#include "Common/Config.h"

#include "ISBI/Visibilities.h"

#include <cstring>

#if defined __AVX__
#include <immintrin.h>
#endif


Visibilities::Visibilities(const ISBI_Parset &ps, unsigned subband)
:
  ps(ps),
  hostVisibilities(boost::extents[ps.nrBaselines()][ps.nrOutputChannelsPerSubband()][ps.nrPolarizations()][ps.nrPolarizations()]),
  subband(subband)
{
  memset(&header, 0, sizeof header);
}


Visibilities &Visibilities::operator += (const Visibilities &other)
{
#if 0
//#if defined __AVX__
  __m256 *dst = reinterpret_cast<__m256 *>(hostVisibilities.origin());
  const __m256 *src = reinterpret_cast<const __m256 *>(other.hostVisibilities.origin());

  for (ssize_t count = hostVisibilities.num_elements() * 2 / 8, i = 0; i < count; i ++)
    dst[i] += src[i];
#else
  for (unsigned channel = 0; channel < ps.nrOutputChannelsPerSubband(); channel ++)
    for (unsigned baseline = 0; baseline < ps.nrBaselines(); baseline ++)
      for (unsigned pol0 = 0; pol0 < ps.nrPolarizations(); pol0++)
        for (unsigned pol1 = 0; pol1 < ps.nrPolarizations(); pol1++)
          hostVisibilities[channel][baseline][pol0][pol1] = other.hostVisibilities[channel][baseline][pol0][pol1]
#endif

  for (unsigned i = 0; i < sizeof(header.weights) / sizeof(header.weights[0]); i ++)
    header.weights[i] += other.header.weights[i];

  startTime = std::min(startTime, other.startTime);
  endTime   = std::max(endTime,   other.endTime);
  return *this;
}


void Visibilities::write(Stream *stream)
{
#if defined USE_LEGACY_VISIBILITIES_FORMAT
  header.magic			 = 0x3B98F002;
#else
  header.magic			 = 0x3B98F003;
#endif
  header.nrReceivers		 = ps.nrStations();
  header.nrPolarizations	 = ps.nrVisibilityPolarizations();
  header.correlationMode	 = ps.correlationMode();
  header.startTime		 = startTime;
  header.endTime		 = endTime;
  header.nrSamplesPerIntegration = ps.nrSamplesPerChannel() * ps.channelIntegrationFactor() * ps.visibilitiesIntegration();
  header.nrChannels		 = ps.nrOutputChannelsPerSubband();
  header.firstChannelFrequency	 = ps.subbandFrequencies().size() > subband ? ps.subbandFrequencies()[subband] - .5 * ps.subbandBandwidth() + .5 * ps.channelBandwidth() /* channel 0 is skipped */ + .5 * ps.outputChannelBandwidth() : 0;
  header.channelBandwidth	 = ps.outputChannelBandwidth();

#if 0
#pragma omp critical (cout)
  {
    for (unsigned count = 0, baseline = 0; baseline < ps.nrBaselines(); baseline ++)
      for (unsigned channel = 0; channel < ps.nrOutputChannelsPerSubband(); channel ++)
	for (unsigned pol = 0; pol < ps.nrVisibilityPolarizations(); pol ++)
	  if (real(hostVisibilities[baseline][channel][pol]) != 0 || imag(hostVisibilities[baseline][channel][pol]) != 0) {
	    std::cout << "hostVisibilities[" << baseline << "][" << channel << "][" << pol << "] = " << hostVisibilities[baseline][channel][pol] << std::endl;

	    if (++ count == 20)
	      goto exit_loop;
	  }
    exit_loop:;
  }
#endif

#if 0
#pragma omp critical (cout)
  {
    unsigned station1 = 0;
    unsigned station2 = 0;
    unsigned baseline = station2 * (station2 + 1) / 2 + station1;
    unsigned pol      = 0;
    std::cout << "visibilities:" << std::endl;

    for (unsigned channel = 0; channel < ps.nrOutputChannelsPerSubband(); channel ++)
      std::cout << channel << ' ' << hostVisibilities[baseline][channel][pol] << std::endl;
  }
#endif

#if 0
//  if (subband == 0)
    for (unsigned channel = 0; channel < ps.nrOutputChannelsPerSubband(); channel ++)
    //for (unsigned channel = 42; channel < 43; channel ++)
      for (unsigned baseline = 0; baseline < 3; baseline ++)
	for (unsigned pol = 0; pol < 4; pol ++)
#pragma omp critical (clog)
	  std::clog << "vis: " << subband << ' ' << channel << ' ' << baseline << ' ' << pol << " = " << hostVisibilities[baseline][channel][pol] << std::endl;
#endif
  stream->write(&header, sizeof(header));
  stream->write(hostVisibilities.origin(), hostVisibilities.bytesize());
}

#ifndef ISBI_VISIBILITES_H
#define ISBI_VISIBILITES_H

#include "ISBI/Parset.h"
//#include "Common/AlignedStdAllocator.h"
#include "Common/CUDA_Support.h"
#include "Common/Stream/Stream.h"

#include <boost/multi_array.hpp>

#undef USE_LEGACY_VISIBILITIES_FORMAT


class Visibilities
{
  public:
    struct Header {
      uint32_t magic;
      uint16_t nrReceivers;
      uint8_t  nrPolarizations;
      uint8_t  correlationMode;
      double  startTime, endTime;
#if defined USE_LEGACY_VISIBILITIES_FORMAT
      uint32_t weights[78]; // Fixed-sized field, independent of #stations!
#else
      uint32_t weights[300]; // Fixed-sized field, independent of #stations!
#endif
      uint32_t nrSamplesPerIntegration;
      uint16_t nrChannels;
      char     pad0[2];
      double   firstChannelFrequency, channelBandwidth;

#if defined USE_LEGACY_VISIBILITIES_FORMAT
      char     pad1[152];
#else
      char     pad1[288];
#endif
    };

    Visibilities(const ISBI_Parset &, unsigned subband);

    void write(Stream *);

    Visibilities &operator += (const Visibilities &);

    const ISBI_Parset			 	 &ps;
    MultiArrayHostBuffer<std::complex<float>, 4> hostVisibilities;
    TimeStamp					 startTime, endTime;
    unsigned					 subband;
    Header					 header;
};

#endif

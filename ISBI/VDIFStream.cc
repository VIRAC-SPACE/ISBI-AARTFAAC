#include "ISBI/VDIFStream.h"
#include "ISBI/VDIFFileStream.h"
#include "ISBI/VDIFSocketStream.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>

VDIFStream::VDIFStream(double sampleRate)
:
  firstHeader(),
  currentHeader(),
  firstHeaderFound(false),
  invalidFrames(0),
  numberOfFrames(0),
  sampleRate(sampleRate),
  dataSize(0),
  headerSize(0)
{
}


VDIFStream::~VDIFStream()
{
}


void VDIFStream::setFirstHeader(const VDIFHeader &header)
{
  firstHeader = header;
  currentHeader = header;
  dataSize = header.dataSize();
  headerSize = header.headerSize();
  firstHeaderFound = true;
}


std::unique_ptr<VDIFStream> createVDIFStream(const std::string &descriptor, double sampleRate, const TimeStamp &startTime, bool realTime)
{
  if (realTime)
    return std::unique_ptr<VDIFStream>(new VDIFSocketStream(descriptor, sampleRate));
  else
    return std::unique_ptr<VDIFStream>(new VDIFFileStream(descriptor, sampleRate, startTime));
}


int64_t VDIFHeader::timestamp(double sample_rate) const {
    std::tm date{};
    date.tm_year = 2000 + ref_epoch / 2 - 1900;
    date.tm_mon = (ref_epoch & 1) ? 6 : 0;
    date.tm_mday = 1;
    date.tm_hour = 0;
    date.tm_min = 0;
    date.tm_sec = 0;

    std::time_t time = timegm(&date);
    auto time_point = std::chrono::system_clock::from_time_t(time);

    auto exact_time = time_point + std::chrono::seconds(sec_from_epoch);
    int64_t out =  static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(exact_time.time_since_epoch()).count() * sample_rate 
           + dataframe_in_second * samplesPerFrame());
    return out;
}

void VDIFHeader::decode2bit(const std::array<char, maxPacketSize>& frame,
                            std::vector<int8_t>& out) const {
  static const std::array<int8_t, 256 * 4> decodeLUT = []() {
    std::array<int8_t, 256 * 4> lut{};
    for (unsigned byte = 0; byte < 256; ++byte) {
      lut[byte * 4 + 0] = DECODER_LEVEL_2BIT[(byte >> 0) & 0x3];
      lut[byte * 4 + 1] = DECODER_LEVEL_2BIT[(byte >> 2) & 0x3];
      lut[byte * 4 + 2] = DECODER_LEVEL_2BIT[(byte >> 4) & 0x3];
      lut[byte * 4 + 3] = DECODER_LEVEL_2BIT[(byte >> 6) & 0x3];
    }
    return lut;
  }();

  const std::size_t payloadBytes = static_cast<std::size_t>(dataSize());
  const uint8_t* data =
      reinterpret_cast<const uint8_t*>(frame.data() + headerSize());

  const std::size_t totalSamples =
      static_cast<std::size_t>(samplesPerFrame()) * numberOfChannels();

  const std::size_t decodedSamples =
      std::min(totalSamples, payloadBytes * 4);

  out.assign(totalSamples, 0); // resize + zero-fill remainder

  const std::size_t fullBytes = decodedSamples / 4;
  const std::size_t tail      = decodedSamples % 4;

  for (std::size_t i = 0; i < fullBytes; ++i) {
    const int8_t* decoded = &decodeLUT[static_cast<std::size_t>(data[i]) * 4];
    std::copy_n(decoded, 4, out.begin() + i * 4);
  }

  if (tail != 0) {
    const int8_t* decoded = &decodeLUT[static_cast<std::size_t>(data[fullBytes]) * 4];
    std::copy_n(decoded, tail, out.begin() + fullBytes * 4);
  }
}

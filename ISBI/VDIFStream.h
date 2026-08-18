#ifndef RADIOBLOCKS_VDIFSTREAM_H
#define RADIOBLOCKS_VDIFSTREAM_H

#include "Common/Stream/Stream.h"
#include "Common/TimeStamp.h"

#include <array>
#include <complex>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

static constexpr int8_t DECODER_LEVEL_2BIT[] = { -3, -1, 1, 3 };
static constexpr uint32_t maxPacketSize = 8032;

enum HeaderStatus {
  INVALID = 0,
  VALID_NOT_START_BLOCK,
  VALID
};

struct VDIFHeader {
  // Word 0
  uint32_t      sec_from_epoch:30;
  uint8_t       legacy_mode:1, invalid:1;
  // Word 1
  uint32_t      dataframe_in_second:24;
  uint8_t       ref_epoch:6, unassiged:2;
  // Word 2
  uint32_t      dataframe_length:24;
  uint8_t       log2_nchan:5, version:3;
  // Word 3
  uint16_t      station_id:16, thread_id:10;
  uint8_t       bits_per_sample:5, data_type:1;
  // Word 4
  uint32_t      user_data1:24;
  uint8_t       edv:8;
  // Word 5-7
  uint32_t      user_data2,user_data3,user_data4;

  
  int64_t timestamp(double sample_rate) const;
  uint32_t frameSize() const;
  uint32_t dataSize() const;
  uint32_t headerSize() const;
  uint32_t samplesPerFrame() const;
  uint32_t numberOfChannels() const;
  void decode2bit(const std::array<char, maxPacketSize>& frame, std::vector<int8_t>& out) const;

  friend std::ostream& operator<<(std::ostream& os, const VDIFHeader& header) {
    os << "----- VDIF HEADER -----" << std::endl;
    os << "sec_from_epoch: " << header.sec_from_epoch << std::endl;
    os << "legacy_mode: " << static_cast<int>(header.legacy_mode) << std::endl;
    os << "invalid: " << static_cast<int>(header.invalid) << std::endl;
    os << "dataframe_in_second: " << header.dataframe_in_second << std::endl;
    os << "ref_epoch: " << static_cast<int>(header.ref_epoch) << std::endl;
    os << "dataframe_length: " << header.dataframe_length << std::endl;
    os << "log2_nchan: " << static_cast<int>(header.log2_nchan) << std::endl;
    os << "version: " << static_cast<int>(header.version) << std::endl;
    os << "station_id: " << header.station_id << std::endl;
    os << "thread_id: " << header.thread_id << std::endl;
    os << "bits_per_sample: " << static_cast<int>(header.bits_per_sample) << std::endl;
    os << "data_type: " << static_cast<int>(header.data_type) << std::endl;
    os << "user_data1: " << header.user_data1 << std::endl;
    os << "edv: " << static_cast<int>(header.edv) << std::endl;
    os << "user_data2: " << header.user_data2 << std::endl;
    os << "user_data3: " << header.user_data3 << std::endl;
    os << "user_data4: " << header.user_data4 << std::endl;
    return os;
  }

};

// Common interface for the offline (file) and real-time (UDP) VDIF sources.
// The first valid frame that is seen defines the frame layout; frames that do
// not match it are dropped by the derived streams.
class VDIFStream : public Stream {
  public:
    virtual ~VDIFStream();

    // Reads one complete VDIF frame into frame, which must have room for
    // maxPacketSize bytes. Returns false when no frame was available before the
    // read timed out (only happens on real-time input); throws
    // EndOfStreamException when no further frames will ever arrive.
    virtual bool read(char *frame) = 0;

    // NOT USED, they come from Stream class.
    size_t tryWrite(const void *ptr, size_t size) { return 0; }
    size_t tryRead(void *ptr, size_t size) { return 0; }

    bool     haveFirstHeader() const { return firstHeaderFound; }
    int64_t  getFirstTimestamp() const;
    const VDIFHeader &getFirstHeader() const { return firstHeader; }
    const VDIFHeader &getCurrentHeader() const { return currentHeader; }
    uint64_t getNumberOfFrames() const { return numberOfFrames; }
    uint64_t getInvalidFrames() const { return invalidFrames; }

  protected:
    VDIFStream(double sampleRate);

    static HeaderStatus checkHeader(const VDIFHeader &header);

    // Records the layout that all following frames must have.
    void setFirstHeader(const VDIFHeader &header);
    bool matchesFirstHeader(const VDIFHeader &header) const;

    VDIFHeader firstHeader, currentHeader;

    bool firstHeaderFound;

    uint64_t invalidFrames;
    uint64_t numberOfFrames;

    double sampleRate;
    uint32_t dataSize;
    uint32_t headerSize;
};

// Creates the stream that matches the run mode: a UDP receiver in real-time
// mode, a file reader otherwise. The descriptor is a filename for file input
// and [udp:][address:]port for real-time input; startTime is only used to seek
// in a file.
std::unique_ptr<VDIFStream> createVDIFStream(const std::string &descriptor, double sampleRate, const TimeStamp &startTime, bool realTime);

inline uint32_t VDIFHeader::headerSize() const {
  return 16 + 16 * (1 - legacy_mode);
}

inline uint32_t VDIFHeader::frameSize() const {
  return 8 * dataframe_length;
}

inline uint32_t VDIFHeader::dataSize() const {
  return 8 * dataframe_length - headerSize();
}

inline uint32_t VDIFHeader::numberOfChannels() const {
  return 1 << log2_nchan;
}

inline uint32_t VDIFHeader::samplesPerFrame() const {
  uint32_t bps = bits_per_sample + 1;
  return dataSize() * 8 / bps / numberOfChannels();
}

inline HeaderStatus VDIFStream::checkHeader(const VDIFHeader &header) {
  const uint32_t *words = reinterpret_cast<const uint32_t *>(&header);

  if (words[0] == 0x11223344 ||
      words[1] == 0x11223344 ||
      words[2] == 0x11223344 ||
      words[3] == 0x11223344) {
    return HeaderStatus::INVALID;
  } else if (header.ref_epoch == 0 && header.sec_from_epoch == 0) {
    return HeaderStatus::INVALID;
  } else if (header.frameSize() <= header.headerSize() || header.frameSize() > maxPacketSize) {
    return HeaderStatus::INVALID;
  }

  return HeaderStatus::VALID;
}

inline bool VDIFStream::matchesFirstHeader(const VDIFHeader &header) const {
  return firstHeaderFound && header.headerSize() == headerSize && header.dataSize() == dataSize;
}

inline int64_t VDIFStream::getFirstTimestamp() const {
  return firstHeaderFound ? firstHeader.timestamp(sampleRate) : 0;
}

#endif //RADIOBLOCKS_VDIFSTREAM_H

#ifndef RADIOBLOCKS_VDIFSTREAM_H
#define RADIOBLOCKS_VDIFSTREAM_H

#include "Common/Stream/FileStream.h"
#include "Common/TimeStamp.h"

#include <fstream>
#include <array>
#include <complex>
#include <vector>
#include <ctime>

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

class VDIFStream : public Stream {
  private:
    std::ifstream file;
    std::vector<char> ioBuffer;

    VDIFHeader firstHeader, currentHeader;

    bool firstHeaderFound;

    uint32_t invalidFrames;
    uint32_t numberOfFrames;

    double sampleRate;
    uint32_t dataSize;
    uint32_t headerSize;

    bool readFirstHeader();
    void findNextValidHeader();
    HeaderStatus checkHeader();

    void atTimestamp(const TimeStamp &ts);
    bool readHeaderAtFrame(uint64_t frameIndex, VDIFHeader &hdr);
  public:
    VDIFStream(std::string inputFile, double sampleRate, TimeStamp startTime);

    void read(char* frame);

    // NOT USED, they come from Stream class.
    size_t tryWrite(const void *ptr, size_t size) { return 0; }
    size_t tryRead(void *ptr, size_t size) { return 0; }

    int64_t getFirstTimestamp() const;
    ~VDIFStream();
};

inline uint32_t VDIFHeader::headerSize() const {
  return 16 + 16 * (1 - legacy_mode);
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

inline HeaderStatus VDIFStream::checkHeader() {
  if (((uint32_t *)&currentHeader)[0] == 0x11223344 ||
      ((uint32_t *)&currentHeader)[1] == 0x11223344 ||
      ((uint32_t *)&currentHeader)[2] == 0x11223344 ||
      ((uint32_t *)&currentHeader)[3] == 0x11223344) {
    return HeaderStatus::INVALID;
  } else if (currentHeader.ref_epoch == 0  && currentHeader.sec_from_epoch == 0) {
    return HeaderStatus::INVALID;
  }

  return HeaderStatus::VALID;
}

inline int64_t VDIFStream::getFirstTimestamp() const {
  return firstHeader.timestamp(sampleRate);
}

#endif //RADIOBLOCKS_VDIFSTREAM_H

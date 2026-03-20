#include "VDIFStream.h"

#include <iostream>
#include <algorithm>
#include <cstdio>
#include <stdlib.h>
#include <cstdint>
#include <stdexcept>
#include <chrono>
#include <vector>
#include <cstring>

constexpr uint32_t HEADER_SIZE = 32; // bytes
constexpr uint32_t DATA_SIZE = 8000; // bytes
constexpr std::size_t READ_BUFFER_SIZE = 1u << 20;

VDIFStream::VDIFStream(std::string inputFile, double sampleRate, TimeStamp startTime) 
  : ioBuffer(READ_BUFFER_SIZE), 
    firstHeaderFound(false), 
    invalidFrames(0), 
    numberOfFrames(0), 
    sampleRate(sampleRate), 
    dataSize(0), 
    headerSize(0) { 

    file.rdbuf()->pubsetbuf(ioBuffer.data(), ioBuffer.size());
    file.open(inputFile, std::ios::binary);
    std::cout << "Created a new VDIFStream object for " << inputFile << std::endl;

    if (!file.is_open()) { throw std::runtime_error("Failed to open " + inputFile + " file!"); }
    if (!readFirstHeader()) { throw std::runtime_error("Could not find a valid header!"); }


    dataSize = firstHeader.dataSize();
    headerSize = firstHeader.headerSize();

    file.clear();
    file.seekg(static_cast<off_t>(numberOfFrames) * (headerSize + dataSize), std::ios::beg);
    if (!file) { throw std::runtime_error("Failed to seek to the first frame!"); }

    atTimestamp(startTime);
}

bool VDIFStream::readHeaderAtFrame(uint64_t frameIndex, VDIFHeader &hdr) {
  const std::streamoff offset =
    static_cast<std::streamoff>(frameIndex) * (headerSize + dataSize);

  file.clear();
  file.seekg(offset, std::ios::beg);
  if (!file) {
    return false;
  }

  file.read(reinterpret_cast<char*>(&hdr), headerSize);
  return file.gcount() == static_cast<std::streamsize>(headerSize);
}

void VDIFStream::atTimestamp(const TimeStamp &ts) {
  const int64_t target = static_cast<int64_t>(ts);
  const int64_t firstTs = firstHeader.timestamp(sampleRate);
  const int64_t samplesPerFrame = firstHeader.samplesPerFrame();

  int64_t estimatedFrame = (target - firstTs) / samplesPerFrame;
  if (estimatedFrame < 0) {
    estimatedFrame = 0;
  }

  uint64_t frame = static_cast<uint64_t>(estimatedFrame);
  VDIFHeader header;

  while (true) {
    if (!readHeaderAtFrame(frame, header)) {
      throw std::runtime_error("VDIFSTream::atTimestamp: target beyond EOF");
    }

    currentHeader = header;
    if (checkHeader() != HeaderStatus::VALID) {
      ++frame;
      continue;
    }

    const int64_t frameStart = header.timestamp(sampleRate);
    const int64_t frameEnd = frameStart + header.samplesPerFrame();

    if (target < frameStart) {
      if (frame == 0) {
        break;
      }
      --frame;
      continue;
    }

    if (target < frameEnd) {
      break;
    }

    ++frame;
  }

  numberOfFrames = frame;
  file.clear();
  file.seekg(static_cast<std::streamoff>(frame) * (headerSize + dataSize), std::ios::beg);

  if (!file) {
    throw std::runtime_error("VDIFStream::atTimestamp: failed final seek");
  }

  std::cout << "Seeked to frame " << frame
    << " timestamp " << currentHeader.timestamp(sampleRate)
    << " for target " << target << std::endl;
}

bool VDIFStream::readFirstHeader() {
  while (file.read(reinterpret_cast<char*>(&currentHeader), HEADER_SIZE)) {
    if (checkHeader() == HeaderStatus::VALID) {
      std::memcpy(&firstHeader, &currentHeader, HEADER_SIZE);
      firstHeaderFound = true;

      const off_t offset = static_cast<off_t>(numberOfFrames) * (HEADER_SIZE + DATA_SIZE);
      std::cout << "Found first valid header at offset: " << offset << std::endl;
      return true;
    }

    ++invalidFrames;
    ++numberOfFrames;
    file.ignore(DATA_SIZE);
  }

  return false;
}

void VDIFStream::read(char* frame) {
  const std::streamsize frameBytes = static_cast<std::streamsize>(headerSize + dataSize);

  file.read(frame, frameBytes);
  if (file.gcount() != frameBytes) {
    if (file.eof()) throw EndOfStreamException("VDIFStream::read EOF reached");
    throw EndOfStreamException("VDIFStream::read incomplete frame read");
  }

  std::memcpy(&currentHeader, frame, headerSize);
  if (checkHeader() != HeaderStatus::VALID) {
    const off_t expectedOffset = static_cast<off_t>(numberOfFrames) * (headerSize + dataSize);
    std::cout << "Invalid header found at offset " << expectedOffset << std::endl;
    findNextValidHeader();

    file.read(frame, frameBytes);
    if (file.gcount() != frameBytes) {
      if (file.eof()) throw EndOfStreamException("VDIFStream::read EOF reached");
      throw EndOfStreamException("VDIFStream::read incomplete frame read");
    }

    std::memcpy(&currentHeader, frame, headerSize);
  }

  numberOfFrames++;
}

void VDIFStream::findNextValidHeader() {
  while (true) {
    numberOfFrames++;

    file.read(reinterpret_cast<char*>(&currentHeader), headerSize);
    if (file.gcount() != static_cast<std::streamsize>(headerSize)) {
      throw EndOfStreamException("VDIFStream::findNextValidHeader: truncated header");
    }

    if (checkHeader() == HeaderStatus::VALID) {
      file.seekg(-static_cast<std::streamoff>(headerSize), std::ios::cur);
      if (!file) {
        throw EndOfStreamException("VDIFStream::findNextValidHeader: seek failed");
      }
      return;
    }

    ++invalidFrames;
    file.ignore(dataSize);
    if (!file) {
      throw EndOfStreamException("VDIFStream::findNextValidHeader: skip failed");
    }
  }
}


VDIFStream::~VDIFStream() {
  std::cout << "Total frames read: " <<  numberOfFrames << std::endl;
  file.close();
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

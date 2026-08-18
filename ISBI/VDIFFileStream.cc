#include "ISBI/VDIFFileStream.h"

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

VDIFFileStream::VDIFFileStream(std::string inputFile, double sampleRate, TimeStamp startTime) 
  : VDIFStream(sampleRate),
    ioBuffer(READ_BUFFER_SIZE) { 

    file.rdbuf()->pubsetbuf(ioBuffer.data(), ioBuffer.size());
    file.open(inputFile, std::ios::binary);
    std::cout << "Created a new VDIFFileStream object for " << inputFile << std::endl;

    if (!file.is_open()) { throw std::runtime_error("Failed to open " + inputFile + " file!"); }
    if (!readFirstHeader()) { throw std::runtime_error("Could not find a valid header!"); }

    file.clear();
    file.seekg(static_cast<off_t>(numberOfFrames) * (headerSize + dataSize), std::ios::beg);
    if (!file) { throw std::runtime_error("Failed to seek to the first frame!"); }

    atTimestamp(startTime);
}

bool VDIFFileStream::readHeaderAtFrame(uint64_t frameIndex, VDIFHeader &hdr) {
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

void VDIFFileStream::atTimestamp(const TimeStamp &ts) {
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
      throw std::runtime_error("VDIFFileStream::atTimestamp: target beyond EOF");
    }

    currentHeader = header;
    if (checkHeader(header) != HeaderStatus::VALID) {
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
    throw std::runtime_error("VDIFFileStream::atTimestamp: failed final seek");
  }

  std::cout << "Seeked to frame " << frame
    << " timestamp " << currentHeader.timestamp(sampleRate)
    << " for target " << target << std::endl;
}

bool VDIFFileStream::readFirstHeader() {
  VDIFHeader header;

  while (file.read(reinterpret_cast<char*>(&header), HEADER_SIZE)) {
    if (checkHeader(header) == HeaderStatus::VALID) {
      setFirstHeader(header);

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

bool VDIFFileStream::read(char* frame) {
  const std::streamsize frameBytes = static_cast<std::streamsize>(headerSize + dataSize);

  file.read(frame, frameBytes);
  if (file.gcount() != frameBytes) {
    if (file.eof()) throw EndOfStreamException("VDIFFileStream::read EOF reached");
    throw EndOfStreamException("VDIFFileStream::read incomplete frame read");
  }

  std::memcpy(&currentHeader, frame, headerSize);
  if (checkHeader(currentHeader) != HeaderStatus::VALID) {
    const off_t expectedOffset = static_cast<off_t>(numberOfFrames) * (headerSize + dataSize);
    std::cout << "Invalid header found at offset " << expectedOffset << std::endl;
    findNextValidHeader();

    file.read(frame, frameBytes);
    if (file.gcount() != frameBytes) {
      if (file.eof()) throw EndOfStreamException("VDIFFileStream::read EOF reached");
      throw EndOfStreamException("VDIFFileStream::read incomplete frame read");
    }

    std::memcpy(&currentHeader, frame, headerSize);
  }

  numberOfFrames++;
  return true;
}

void VDIFFileStream::findNextValidHeader() {
  while (true) {
    numberOfFrames++;

    file.read(reinterpret_cast<char*>(&currentHeader), headerSize);
    if (file.gcount() != static_cast<std::streamsize>(headerSize)) {
      throw EndOfStreamException("VDIFFileStream::findNextValidHeader: truncated header");
    }

    if (checkHeader(currentHeader) == HeaderStatus::VALID) {
      file.seekg(-static_cast<std::streamoff>(headerSize), std::ios::cur);
      if (!file) {
        throw EndOfStreamException("VDIFFileStream::findNextValidHeader: seek failed");
      }
      return;
    }

    ++invalidFrames;
    file.ignore(dataSize);
    if (!file) {
      throw EndOfStreamException("VDIFFileStream::findNextValidHeader: skip failed");
    }
  }
}


VDIFFileStream::~VDIFFileStream() {
  std::cout << "Total frames read: " <<  numberOfFrames << std::endl;
  file.close();
}

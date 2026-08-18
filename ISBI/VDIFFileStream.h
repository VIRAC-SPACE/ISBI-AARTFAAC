#ifndef RADIOBLOCKS_VDIFFILESTREAM_H
#define RADIOBLOCKS_VDIFFILESTREAM_H

#include "ISBI/VDIFStream.h"
#include "Common/TimeStamp.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// Reads VDIF frames from a recording. Used when the correlator does not run in
// real-time mode.
class VDIFFileStream : public VDIFStream {
  private:
    std::ifstream file;
    std::vector<char> ioBuffer;

    bool readFirstHeader();
    void findNextValidHeader();

    void atTimestamp(const TimeStamp &ts);
    bool readHeaderAtFrame(uint64_t frameIndex, VDIFHeader &hdr);
  public:
    VDIFFileStream(std::string inputFile, double sampleRate, TimeStamp startTime);
    ~VDIFFileStream();

    bool read(char *frame);
};

#endif //RADIOBLOCKS_VDIFFILESTREAM_H

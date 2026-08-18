#ifndef RADIOBLOCKS_VDIFSOCKETSTREAM_H
#define RADIOBLOCKS_VDIFSOCKETSTREAM_H

#include "ISBI/VDIFStream.h"
#include "Common/Stream/SocketStream.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

// Receives one complete VDIF frame per UDP datagram. Used when the correlator
// runs in real-time mode.
class VDIFSocketStream : public VDIFStream {
  public:
    // descriptor is [udp:][address:]port; an omitted or empty address binds to
    // all interfaces. A read that finds no datagram within readTimeout seconds
    // returns false instead of blocking forever, so that the input thread can
    // notice that the observation ended.
    VDIFSocketStream(const std::string &descriptor, double sampleRate, double readTimeout = 0.1, int receiveBufferSize = 64 * 1024 * 1024);
    ~VDIFSocketStream();

    bool read(char *frame);

    uint64_t getMalformedFrames() const { return malformedFrames; }
    uint64_t getGapsOrReorders() const { return gapsOrReorders; }

  private:
    static void parseDescriptor(const std::string &descriptor, std::string &address, uint16_t &port);

    void setReceiveTimeout(double seconds);
    int  setReceiveBufferSize(int size);
    void updateContinuity(const VDIFHeader &header);

    std::unique_ptr<SocketStream> socket;
    std::string descriptor;

    uint64_t malformedFrames;
    uint64_t gapsOrReorders;
    uint64_t receivedBytes;

    // Last (second, frame in second) seen per VDIF thread, to detect loss.
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> previousFrame;
};

#endif //RADIOBLOCKS_VDIFSOCKETSTREAM_H

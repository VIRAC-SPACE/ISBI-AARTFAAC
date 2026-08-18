#include "ISBI/VDIFSocketStream.h"

#include "Common/SystemCallException.h"

#include <sys/socket.h>
#include <sys/types.h>

#include <boost/lexical_cast.hpp>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

VDIFSocketStream::VDIFSocketStream(const std::string &descriptor, double sampleRate, double readTimeout, int receiveBufferSize)
:
  VDIFStream(sampleRate),
  descriptor(descriptor),
  malformedFrames(0),
  gapsOrReorders(0),
  receivedBytes(0)
{
  std::string address;
  uint16_t port;

  parseDescriptor(descriptor, address, port);

  socket.reset(new SocketStream(address, port, SocketStream::UDP, SocketStream::Server));

  const int obtainedBufferSize = setReceiveBufferSize(receiveBufferSize);
  setReceiveTimeout(readTimeout);

  std::cout << "Created a new VDIFSocketStream object listening on " << address << ':' << port
            << " (receive buffer " << obtainedBufferSize << " bytes)" << std::endl;
}


VDIFSocketStream::~VDIFSocketStream()
{
  std::cout << "VDIFSocketStream " << descriptor
            << ": received " << numberOfFrames << " valid frames ("
            << receivedBytes << " bytes), dropped " << invalidFrames
            << " invalid and " << malformedFrames << " malformed datagrams, "
            << gapsOrReorders << " gaps or reorders" << std::endl;

  // ~SocketStream() may throw, which must not escape from a destructor that is
  // run while another exception is being handled.
  try {
    socket.reset();
  } catch (std::exception &ex) {
    std::cerr << "VDIFSocketStream: failed to close socket: " << ex.what() << std::endl;
  }
}


// Accepts port, address:port and udp:address:port, where an empty address means
// all interfaces. The address is passed to getaddrinfo(), so a hostname works
// as well as a numeric address.
void VDIFSocketStream::parseDescriptor(const std::string &descriptor, std::string &address, uint16_t &port)
{
  std::string remainder = descriptor;

  if (remainder.compare(0, 4, "udp:") == 0)
    remainder = remainder.substr(4);

  const size_t colon = remainder.find_last_of(':');
  std::string portString;

  if (colon == std::string::npos) {
    portString = remainder;
  } else {
    address = remainder.substr(0, colon);
    portString = remainder.substr(colon + 1);
  }

  if (address.empty() || address == "*")
    address = "0.0.0.0";

  try {
    port = boost::lexical_cast<uint16_t>(portString);
  } catch (boost::bad_lexical_cast &) {
    throw std::runtime_error("VDIFSocketStream: cannot parse port from input descriptor \'" + descriptor + "\', expected [udp:][address:]port");
  }

  if (port == 0)
    throw std::runtime_error("VDIFSocketStream: port 0 in input descriptor \'" + descriptor + '\'');
}


int VDIFSocketStream::setReceiveBufferSize(int size)
{
  // A large socket buffer bridges the scheduling gaps of the input thread; the
  // kernel silently clamps the request to net.core.rmem_max.
  if (setsockopt(socket->fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof size) < 0)
    throw SystemCallException("setsockopt(SO_RCVBUF)", errno);

  int obtained = 0;
  socklen_t length = sizeof obtained;

  if (getsockopt(socket->fd, SOL_SOCKET, SO_RCVBUF, &obtained, &length) < 0)
    throw SystemCallException("getsockopt(SO_RCVBUF)", errno);

  return obtained;
}


void VDIFSocketStream::setReceiveTimeout(double seconds)
{
  if (seconds <= 0)
    return;

  struct timeval tv;
  tv.tv_sec  = static_cast<time_t>(seconds);
  tv.tv_usec = static_cast<suseconds_t>((seconds - std::floor(seconds)) * 1e6);

  if (setsockopt(socket->fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) < 0)
    throw SystemCallException("setsockopt(SO_RCVTIMEO)", errno);
}


void VDIFSocketStream::updateContinuity(const VDIFHeader &header)
{
  const std::pair<uint32_t, uint32_t> current(header.sec_from_epoch, header.dataframe_in_second);
  const auto previous = previousFrame.find(header.thread_id);

  if (previous != previousFrame.end()) {
    const std::pair<uint32_t, uint32_t> &last = previous->second;
    const bool nextInSecond      = current.first == last.first && current.second == last.second + 1;
    const bool firstInNextSecond = current.first == last.first + 1 && current.second == 0;

    if (!nextInSecond && !firstInNextSecond)
      ++gapsOrReorders;
  }

  previousFrame[header.thread_id] = current;
}


bool VDIFSocketStream::read(char *frame)
{
  for (;;) {
    // MSG_TRUNC reports the real datagram size, so that oversized datagrams are
    // recognised instead of silently used truncated.
    ssize_t received = recv(socket->fd, frame, maxPacketSize, MSG_TRUNC);

    if (received < 0) {
      if (errno == EINTR)
        continue;

      if (errno == EAGAIN || errno == EWOULDBLOCK)
        return false; // no data for now; let the caller handle what it has

      throw SystemCallException("recv", errno);
    }

    if (received < static_cast<ssize_t>(sizeof(VDIFHeader)) || received > static_cast<ssize_t>(maxPacketSize)) {
      ++malformedFrames;
      continue;
    }

    VDIFHeader header;
    std::memcpy(&header, frame, sizeof header);

    if (checkHeader(header) != HeaderStatus::VALID || header.frameSize() != static_cast<uint32_t>(received)) {
      ++invalidFrames;
      continue;
    }

    if (!firstHeaderFound) {
      setFirstHeader(header);

      std::cout << "VDIFSocketStream " << descriptor << ": first frame received" << std::endl
                << header
                << "samples/frame: " << header.samplesPerFrame() << std::endl;
    } else if (!matchesFirstHeader(header)) {
      // The ring buffer is filled assuming a constant number of samples per
      // frame, so a frame of a different shape cannot be used.
      ++malformedFrames;
      continue;
    }

    currentHeader = header;
    updateContinuity(header);

    ++numberOfFrames;
    receivedBytes += static_cast<uint64_t>(received);
    return true;
  }
}

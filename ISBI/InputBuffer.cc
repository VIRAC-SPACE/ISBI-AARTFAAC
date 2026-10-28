#include "Common/Config.h"
#include "Common/Stream/Descriptor.h"
#include "Common/Affinity.h"

#include "ISBI/InputBuffer.h"
#include "ISBI/VDIFStream.h"

#include <byteswap.h>
#include <omp.h>
#include <sys/socket.h>

#if defined __AVX__
#include <immintrin.h>
#endif

#include <cassert>
#include <memory>
#include <algorithm>
#include <chrono>
#include <vector>

#undef FAKE_TIMES
#undef USE_RECVMMSG

#define ISBI_DELAYS

volatile std::sig_atomic_t InputBuffer::signalCaught = false;

namespace {
  const std::array<int8_t, 256 * 4> decodeLUT = [] {
    std::array<int8_t, 256 * 4> lut{};
    for (unsigned byte = 0; byte < 256; ++byte) {
      lut[4 * byte + 0] = DECODER_LEVEL_2BIT[(byte >> 0) & 0x3];
      lut[4 * byte + 1] = DECODER_LEVEL_2BIT[(byte >> 2) & 0x3];
      lut[4 * byte + 2] = DECODER_LEVEL_2BIT[(byte >> 4) & 0x3];
      lut[4 * byte + 3] = DECODER_LEVEL_2BIT[(byte >> 6) & 0x3];
    }
    return lut;
  } ();

  // `step` is nchan * decimation: one sample-time is nchan 2-bit values, and a
  // decimated station simply skips (decimation - 1) of every decimation of them.
  inline void decodeMappedChannelSamples(
      int8_t *__restrict dst,
      unsigned nrSamples,
      const uint8_t *__restrict payload,
      size_t payloadBytes,
      size_t step,
      size_t dataIndex)
  {
    for (unsigned sample = 0; sample < nrSamples; ++sample, dataIndex += step) {
      const size_t byteIndex = dataIndex >> 2;

      if (byteIndex >= payloadBytes) {
        dst[sample] = 0;
        continue;
      }

      const size_t lutIndex = (static_cast<size_t>(payload[byteIndex]) << 2) + (dataIndex & 0x3);
      dst[sample] = decodeLUT[lutIndex];
    }
  }
}

void uncached_memcpy(void *__restrict dst, const void *__restrict src, size_t size)
{
#if defined __AVX__
  size_t done = 0;

#if 0
  // size should be multiple of 2

  while (done < size && ((ptrdiff_t) ((char *) dst + done) & 0x1F) != 0) {
    * (short *) ((char *) dst + done) = * (short *) ((const char *) src + done);
    done += sizeof(short);
  }
#endif

  while (done + sizeof(__m256i) <= size) {
    _mm256_stream_si256((__m256i *) ((char *) dst + done), _mm256_loadu_si256((const __m256i *) ((const char *) src + done)));
    done += sizeof(__m256i);
  }

#if 0
  while (done < size) {
    * (short *) ((char *) dst + done) = * (short *) ((const char *) src + done);
    done += sizeof(short);
  }
#endif
#else
  memcpy(dst, src, size);
#endif
}


void uncached_memclear(void *dst, size_t size)
{
#if defined __AVX__
  size_t done = 0;

#if 0
  // size should be multiple of sizeof(int)

  while (done < size && ((ptrdiff_t) ((char *) dst + done) & 0x1F) != 0) {
    * (int *) ((char *) dst + done) = 0;
    done += sizeof(int);
  }
#endif

  while (done + sizeof(__m256i) <= size) {
    _mm256_stream_si256((__m256i *) ((char *) dst + done), _mm256_setzero_si256());
    done += sizeof(__m256i);
  }

#if 0
  while (done < size) {
    * (int *) ((char *) dst + done) = 0;
    done += sizeof(int);
  }
#endif
#else
  memset(dst, 0, size);
#endif
}


std::ostream &operator << (std::ostream &os, std::function<std::ostream & (std::ostream &os)> function)
{
  return function(os);
}


std::function<std::ostream & (std::ostream &)> InputBuffer::logMessage() const{
  return [&] (std::ostream &os) -> std::ostream & { return os << "InputBuffer " << myFirstStation << '-' << myFirstStation + myNrStations - 1; };
}



InputBuffer::InputBuffer(const ISBI_Parset &ps, MultiArrayHostBuffer<char, 4> hostRingBuffer[], unsigned myFirstSubband, unsigned myNrSubbands, unsigned myFirstStation, unsigned myNrStations, unsigned samplesPerFrameOverride)
:
  ps(ps),
  myFirstSubband(myFirstSubband),
  myNrSubbands(myNrSubbands),
  myFirstStation(myFirstStation),
  myNrStations(myNrStations),
  nrRingBufferSamplesPerSubband(ps.nrRingBufferSamplesPerSubband()),
  samplesPerFrameOverride(samplesPerFrameOverride),
  decimation(ps.decimation(myFirstStation)),
  nrHistorySamples((NR_TAPS - 1) * ps.nrChannelsPerSubbandBeforeFilter()),
  mappedChannels([&] {
    std::vector<uint32_t> channels(myNrSubbands * ps.nrPolarizations());
    const std::vector<ISBI_Parset::ChannelSelector> &channelMapping = ps.channelMapping(myFirstStation);

    // The mapping is per (subband, polarisation) and is backend specific, so it
    // has to cover every (subband, polarisation) this buffer will decode.
    if (channelMapping.size() < (myFirstSubband + myNrSubbands) * ps.nrPolarizations())
      throw std::runtime_error("channelMapping has " + std::to_string(channelMapping.size()) +
                               " entries but " +
                               std::to_string((myFirstSubband + myNrSubbands) * ps.nrPolarizations()) +
                               " are needed for station " + std::to_string(myFirstStation) + ", " + std::to_string(myNrSubbands) +
                               " subbands x " + std::to_string(ps.nrPolarizations()) +
                               " polarisations; pass --channelMapping");

    for (unsigned subband = 0; subband < myNrSubbands; ++subband) {
      const unsigned subbandOffset = (myFirstSubband + subband) * ps.nrPolarizations();

      for (unsigned pol = 0; pol < ps.nrPolarizations(); ++pol) {
        channels[subband * ps.nrPolarizations() + pol] = channelMapping[subbandOffset + pol].channel;
      }
    }

    return channels;
  }()),
  ringBufferBases([&] {
    std::vector<int8_t *> bases(myNrSubbands * ps.nrPolarizations());

    for (unsigned subband = 0; subband < myNrSubbands; ++subband) {
      for (unsigned pol = 0; pol < ps.nrPolarizations(); ++pol) {
        bases[subband * ps.nrPolarizations() + pol] =
            reinterpret_cast<int8_t *>(
                hostRingBuffer[myFirstSubband + subband][myFirstStation][pol][0].origin());
      }
    }

    return bases;
  }()),
  slotsPerThread([&] {
    const std::vector<ISBI_Parset::ChannelSelector> &channelMapping = ps.channelMapping(myFirstStation);
    std::vector<std::vector<uint32_t>> slots;

    for (unsigned subband = 0; subband < myNrSubbands; ++subband) {
      const unsigned subbandOffset = (myFirstSubband + subband) * ps.nrPolarizations();

      for (unsigned pol = 0; pol < ps.nrPolarizations(); ++pol) {
        const unsigned thread = channelMapping[subbandOffset + pol].thread;

        if (thread >= slots.size())
          slots.resize(thread + 1);

        slots[thread].push_back(subband * ps.nrPolarizations() + pol);
      }
    }

    return slots;
  }()),
  hostRingBuffer(hostRingBuffer),
  latestWriteTime(0, ps.clockSpeed()),
  stop(false),
  readerAndWriterSynchronization(nrRingBufferSamplesPerSubband, ps.startTime() - nrHistorySamples - ps.maxDelay()),
  inputThread(&InputBuffer::inputThreadBody, this),
  logThread(&InputBuffer::logThreadBody, this),
  noInputThreadPtr(ps.realTime() ? new std::thread(&InputBuffer::noInputThreadBody, this) : 0)
{
#if defined __AVX__
  // check alignment for _mm256_stream_si256
  assert((ptrdiff_t) hostRingBuffer[0].origin() % 32 == 0);
  std::cout << "myNrStations " <<  myNrStations << std::endl;
  //assert(myNrStations * ps.nrPolarizations() * ps.nrBytesPerComplexSample() % 32 == 0);
#endif

#pragma omp critical (clog)
  std::clog << logMessage() << " created by CPU " << currentCPU() << " on node " << currentNode() << ", memory at node " << node(hostRingBuffer) << std::endl;
}

InputBuffer::~InputBuffer()
{
  readerAndWriterSynchronization.noMoreReading(); // FIXME: not here
  stop = true;

  if (noInputThreadPtr.get() != nullptr)
    noInputThreadPtr->join();

  logThread.join();
  inputThread.join();
}


void InputBuffer::handleConsecutivePackets(std::array<std::array<char, maxPacketSize>, maxNrPacketsInBuffer>& packetBuffer, unsigned firstPacket, unsigned lastPacket) {
  // The run may hold several VDIF threads, and a station that splits its
  // channels over threads emits them interleaved and not always in a fixed
  // order (n26l2's Cm: thread 1, 0, 0, 1, 1, 0 over timestamps t, t, t+1, ...).
  // So the run is bounded by the smallest and largest timestamp it contains,
  // and every frame is placed by its OWN timestamp rather than by its position,
  // which costs nothing for the single-thread case and is the only thing that
  // works for the multi-thread one.
  TimeStamp beginTime(0, ps.clockSpeed()), endTime(0, ps.clockSpeed());

  for (unsigned packet = firstPacket; packet < lastPacket; ++packet) {
    const VDIFHeader* packetHeader = reinterpret_cast<const VDIFHeader*>(packetBuffer[packet].data());
    TimeStamp packetTime(frameTimestamp(packetHeader), ps.clockSpeed());
    TimeStamp packetEnd(packetTime + samplesPerFrame(packetHeader));

    if (packet == firstPacket || packetTime < beginTime)
      beginTime = packetTime;

    if (packet == firstPacket || packetEnd > endTime)
      endTime = packetEnd;
  }

  std::lock_guard<std::mutex> latestWriteTimeLock(latestWriteTimeMutex);

  if (beginTime >= latestWriteTime) {
    latestWriteTime = endTime;

    readerAndWriterSynchronization.startWrite(beginTime, endTime);

    for (unsigned packet = firstPacket; packet < lastPacket; ++packet) {
      const VDIFHeader* packetHeader = reinterpret_cast<const VDIFHeader*>(packetBuffer[packet].data());
      const uint8_t *payload = reinterpret_cast<const uint8_t*>(packetBuffer[packet].data() + packetHeader->headerSize());
      const size_t payloadBytes = packetHeader->dataSize();
      const size_t step = (size_t) packetHeader->numberOfChannels() * decimation;
      const unsigned nrTimes = samplesPerFrame(packetHeader);
      const unsigned thread = packetHeader->thread_id;

      // A thread this station was not told to use carries nothing we want.
      if (thread >= slotsPerThread.size())
        continue;

      TimeStamp packetTime(frameTimestamp(packetHeader), ps.clockSpeed());
      const unsigned timeIndex = packetTime % nrRingBufferSamplesPerSubband;
      const unsigned firstSpan = std::min(nrTimes, nrRingBufferSamplesPerSubband - timeIndex);
      const unsigned secondSpan = nrTimes - firstSpan;

      for (uint32_t slot : slotsPerThread[thread]) {
        const size_t mappedIndex = mappedChannels[slot];
        int8_t *ringBuffer = ringBufferBases[slot];

        decodeMappedChannelSamples(ringBuffer + timeIndex, firstSpan, payload, payloadBytes, step, mappedIndex);

        if (secondSpan > 0) {
          decodeMappedChannelSamples(
              ringBuffer,
              secondSpan,
              payload,
              payloadBytes,
              step,
              mappedIndex + static_cast<size_t>(firstSpan) * step);
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(validDataMutex);
      validData.exclude(TimeStamp(0, 1), endTime - nrRingBufferSamplesPerSubband);
      const SparseSet<TimeStamp>::Ranges &ranges = validData.getRanges();

      if (ranges.size() < 16 || ranges.back().end == beginTime) {
        validData.include(beginTime, endTime);
      }
    }

    readerAndWriterSynchronization.finishedWrite(endTime);
  }
}




void InputBuffer::inputThreadBody() {
  TimeStamp expectedTimeStamp(0, ps.clockSpeed()), stopTime = ps.stopTime() + ps.nrSamplesPerSubbandBeforeFilter();
#if defined FAKE_TIMES
  //expectedTimeStamp = ps.startTime() - nrHistorySamples - 20;
  expectedTimeStamp = TimeStamp::now(ps.clockSpeed()) - nrHistorySamples - 20;
#pragma omp critical (clog)
  std::clog<<"expectedTimeStamp " << expectedTimeStamp << std::endl;

#endif

  TimeStamp seekTime = ps.startTime() - nrHistorySamples - ps.maxDelay();

  // In real-time mode the frames arrive over UDP, otherwise they are read from
  // a recording that is seeked to the start of the observation.
  std::unique_ptr<VDIFStream> vdifStream = createVDIFStream(ps.inputDescriptors()[myFirstStation], ps.sampleRate(), seekTime, ps.realTime(), decimation);

#pragma omp critical (clog)
  {
    std::clog << "Station " << myFirstStation << ' ' << (ps.realTime() ? "real-time" : "file") << " input " << ps.inputDescriptors()[myFirstStation] << ", first VDIF timestamp: ";

    if (vdifStream->haveFirstHeader())
      std::clog << vdifStream->getFirstTimestamp() << " samples";
    else
      std::clog << "not known yet, no frame received";

    std::clog << " vs ps.startTime()=" << ps.startTime() << std::endl;
  }

  std::array<std::array<char, maxPacketSize>, maxNrPacketsInBuffer> packetBuffer;

  bool printedImpossibleTimeStampWarning = false;
  unsigned nrPackets, firstPacket, nextPacket;
  TimeStamp timeStamp(0, ps.clockSpeed()), previousTimeStamp(0, ps.clockSpeed()); 

#if defined USE_RECVMMSG
  struct iovec   iovecs[maxNrPacketsInBuffer];
  struct mmsghdr msgvec[maxNrPacketsInBuffer];

  for (int packet = 0; packet < maxNrPacketsInBuffer; packet ++) {
    iovecs[packet].iov_base		  = packetBuffer[packet];
    iovecs[packet].iov_len		  = maxPacketSize;
    msgvec[packet].msg_hdr.msg_iov    = &iovecs[packet];
    msgvec[packet].msg_hdr.msg_iovlen = 1;
  }
#endif

  do {
    //#if defined USE_RECVMMSG  
    try {
      // One batched call: for UDP this is a single recvmmsg(), so the socket
      // is drained far faster than with one recv() per frame. Returns 0 when
      // the real-time input is idle; handle whatever was received so far.
      nrPackets = vdifStream->readMany(packetBuffer.data(), maxNrPacketsInBuffer);
    }
    catch (Stream::EndOfStreamException) {
#pragma omp critical (clog)
      std::clog <<  logMessage()  << " caught EndOfStreamException" << std::endl;
      stop = true;
    } 

    /* catch (const SystemCallException &ex) {
       if (ex.error != EAGAIN && ex.error != EINTR) // expect timeout; rethrow others
       throw;
       }*/

    for (firstPacket = nextPacket = 0; nextPacket < nrPackets; nextPacket ++) {
      const VDIFHeader* header = reinterpret_cast<const VDIFHeader*>(packetBuffer[nextPacket].data());
      timeStamp = TimeStamp(frameTimestamp(header), ps.clockSpeed());

      // A frame continues the current run if it advances by one frame, or if it
      // REPEATS the previous timestamp -- which is what a second VDIF thread of
      // the same station looks like, since every thread stamps the same time.
      if (timeStamp != expectedTimeStamp && timeStamp != previousTimeStamp) {
        if (firstPacket < nextPacket) {
          handleConsecutivePackets(packetBuffer, firstPacket, nextPacket);
        }

        if (ps.realTime() && abs(TimeStamp::now(ps.clockSpeed()) - timeStamp) > 15 * ps.subbandBandwidth()) {
          if (!printedImpossibleTimeStampWarning) {
            printedImpossibleTimeStampWarning = true;
#pragma omp critical (clog)
            std::clog << logMessage() << ": impossible timestamp " << timeStamp << std::endl;
          }

          firstPacket = nextPacket + 1;
          timeStamp = 0;
        } else {
          printedImpossibleTimeStampWarning = false;
          firstPacket = nextPacket;
        }
      }

      previousTimeStamp = timeStamp;
      expectedTimeStamp = timeStamp + samplesPerFrame(header);
    }


    if (firstPacket < nextPacket) {
      handleConsecutivePackets(packetBuffer, firstPacket, nextPacket);
    } 


    //#endif
  } while (timeStamp < stopTime && !stop && !signalCaught);

  readerAndWriterSynchronization.noMoreWriting();

#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  /*  } catch (std::exception &ex) {
      readerAndWriterSynchronization.noMoreWriting();


#pragma omp critical (cerr)
std::cerr << logMessage() << " caught std::exception: " << ex.what() << std::endl;
}*/

#endif
}


void InputBuffer::logThreadBody()
{
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  try {
#endif
    while (!stop && !signalCaught) {
      std::this_thread::sleep_for(std::chrono::seconds(1));

      std::lock_guard<std::mutex> lock(validDataMutex);
#pragma omp critical (clog)
      std::clog << logMessage() << ", valid: " << validData << std::endl;
    }
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  } catch (std::exception &ex) {
#pragma omp critical (cerr)
    std::cerr << logMessage() << "caught std::exception: " << ex.what() << std::endl;
  }
#endif
}



void InputBuffer::noInputThreadBody()
{
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  try {
#endif
    // forces the correlator to proceed even if no data is received

    WallClockTime wallClock;
    bool lateLastTime = false;

    for (TimeStamp timeStamp = ps.startTime(); timeStamp < ps.stopTime() + ps.nrSamplesPerSubbandBeforeFilter() * ps.visibilitiesIntegration() && !stop && !signalCaught; timeStamp += ps.subbandBandwidth() / 10) {
      // Wait well past timeStamp before deciding that no data will arrive.
      // This used to be subbandBandwidth()/3 (0.333 s), which is far tighter
      // than the jitter of the input thread at full rate. Overshooting it is
      // not a benign miss: forcing sets latestWriteTime = timeStamp, and
      // handleConsecutivePackets() only writes when beginTime >=
      // latestWriteTime, so every subsequent real batch is silently discarded
      // and the stream never recovers ("forcing ..." with no matching
      // "resumed normal operation"). The tolerance must therefore exceed
      // normal input jitter; it stays well inside the ring buffer, which
      // holds nrRingBufferSamplesPerSubband (~4 s by default).
      wallClock.waitUntil(timeStamp + 2 * ps.subbandBandwidth());

      std::lock_guard<std::mutex> lock(latestWriteTimeMutex);

      if (latestWriteTime < timeStamp) {
	readerAndWriterSynchronization.startWrite(latestWriteTime, timeStamp);
	readerAndWriterSynchronization.finishedWrite(timeStamp);
	latestWriteTime = timeStamp;

	if (!lateLastTime) {
#pragma omp critical (clog)
	  std::clog << logMessage() << ": forcing correlator to continue without data " << timeStamp << std::endl;
	  lateLastTime = true;
	}
      } else if (lateLastTime) {
#pragma omp critical (clog)
	std::clog << logMessage() << ": resumed normal operation " << timeStamp << std::endl;
	lateLastTime = false;
      }
    }

    readerAndWriterSynchronization.noMoreWriting();
#if !defined CREATE_BACKTRACE_ON_EXCEPTION
  } catch (std::exception &ex) {
#pragma omp critical (cerr)
    std::cerr << logMessage() << " caught std::exception: " << ex.what() << std::endl;

    readerAndWriterSynchronization.noMoreWriting();
  }
#endif
}



SparseSet<TimeStamp> InputBuffer::getCurrentValidData(const TimeStamp &earlyStartTime, const TimeStamp &endTime)
{
  std::lock_guard<std::mutex> lock(validDataMutex);
  return validData & SparseSet<TimeStamp>(earlyStartTime, endTime);
}

void InputBuffer::fillInMissingSamples(const TimeStamp &startTime, unsigned subband, SparseSet<TimeStamp> &validData)
{
  TimeStamp earlyStartTime   = startTime - nrHistorySamples - ps.maxDelay();
  TimeStamp endTime          = startTime + ps.nrSamplesPerSubbandBeforeFilter() + ps.maxDelay();

  validData = getCurrentValidData(earlyStartTime, endTime);
  SparseSet<TimeStamp> flaggedData = validData.invert(earlyStartTime, endTime);
  const SparseSet<TimeStamp>::Ranges &flaggedRanges = flaggedData.getRanges();
  size_t size = myNrStations * ps.nrBytesPerRealSample(); 
  
  for (const SparseSet<TimeStamp>::range &it : flaggedRanges) {
    for (unsigned timeIndex = it.begin % nrRingBufferSamplesPerSubband, timeEndIndex = it.end % nrRingBufferSamplesPerSubband; timeIndex != timeEndIndex;) {
      for (unsigned pol = 0; pol < ps.nrPolarizations(); pol++) {
        uncached_memclear(hostRingBuffer[subband][myFirstStation][pol][timeIndex].origin(), size);
      }

      if (++ timeIndex == nrRingBufferSamplesPerSubband)
        timeIndex = 0;
    }
  }

  unsigned nrHistorySamples = (NR_TAPS - 1) * ps.nrChannelsPerSubbandBeforeFilter();
  unsigned nrSamples        = nrHistorySamples + ps.nrSamplesPerSubbandBeforeFilter();

  if (subband == 0)
#pragma omp critical (clog)
    std::clog << logMessage() << ' ' << earlyStartTime << " flagged: " << 100.0 * (int64_t) flaggedData.count() / nrSamples << "% (" << flaggedRanges.size() << ')' << std::endl;
}


void InputBuffer::startReadTransaction(const TimeStamp &startTime)
{
  TimeStamp earlyStartTime   = startTime - nrHistorySamples - ps.maxDelay();
  TimeStamp endTime          = startTime + ps.nrSamplesPerSubbandBeforeFilter() + ps.maxDelay();

  readerAndWriterSynchronization.startRead(earlyStartTime, endTime);
}


void InputBuffer::endReadTransaction(const TimeStamp &startTime)
{
  TimeStamp endTime          = startTime + ps.nrSamplesPerSubbandBeforeFilter() + ps.maxDelay();

  readerAndWriterSynchronization.finishedRead(endTime);
}


void InputBuffer::caughtSignal()
{
  signalCaught = true;
}

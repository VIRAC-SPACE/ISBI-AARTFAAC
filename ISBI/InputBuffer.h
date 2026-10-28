#ifndef RADIOBLOCKS_INPUTBUFFER_H
#define RADIOBLOCKS_INPUTBUFFER_H

#include "ISBI/Parset.h"
#include "Common/CUDA_Support.h"
#include "Common/ReaderWriterSynchronization.h"
#include "Common/SparseSet.h"
#include "Common/TimeStamp.h"
#include "ISBI/VDIFStream.h"

#include <boost/multi_array.hpp>

#include <cstdint>
#include <atomic>
#include <csignal>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>


class InputBuffer{
public:
    InputBuffer(const ISBI_Parset &, MultiArrayHostBuffer<char, 4> hostRingBuffer[], unsigned myFirstSubband, unsigned myNrSubbands, unsigned myFirstStation, unsigned myNrStations, unsigned samplesPerFrameOverride);
    ~InputBuffer();

    void fillInMissingSamples(const TimeStamp &time, unsigned subband, SparseSet<TimeStamp> &validData);

    void startReadTransaction(const TimeStamp &);
    void endReadTransaction(const TimeStamp &);

    static void caughtSignal();

private:
    const static unsigned	maxNrPacketsInBuffer = 64;
    const static unsigned	maxPacketSize	     = 8032; // this must not be a power of 2, or performance will collapse due to limited cache associativity

    void inputThreadBody(), noInputThreadBody(), logThreadBody();
    std::function<std::ostream & (std::ostream &)> logMessage() const;

    // A run of frames whose timestamps are contiguous. The frames may belong to
    // several VDIF threads and need not be ordered, so each frame is placed by
    // its own timestamp rather than by its position in the run.
    void handleConsecutivePackets(std::array<std::array<char, maxPacketSize>, maxNrPacketsInBuffer>& packetBuffer, unsigned firstPacket, unsigned lastPacket);

    // Sample-times this frame contributes to the ring buffer: the frame's own
    // payload count (or the parset override), divided by the decimation, since
    // a decimated station keeps only one input sample in every `decimation`.
    unsigned samplesPerFrame(const VDIFHeader *header) const
      { return (samplesPerFrameOverride != 0 ? samplesPerFrameOverride : header->samplesPerFrame()) / decimation; }

    // The frame's timestamp expressed in those same decimated samples, so it
    // indexes the ring buffer correctly.
    int64_t frameTimestamp(const VDIFHeader *header) const
      { return header->timestamp(ps.sampleRate(), samplesPerFrame(header)); }

    SparseSet<TimeStamp> getCurrentValidData(const TimeStamp &earlyStartTime, const TimeStamp &endTime);

    const ISBI_Parset	&ps;
    unsigned			myFirstSubband, myNrSubbands, myFirstStation, myNrStations, nrRingBufferSamplesPerSubband, samplesPerFrameOverride, decimation, nrHistorySamples;
    std::vector<uint32_t>	mappedChannels;
    std::vector<int8_t *>	ringBufferBases;
    // Indices into mappedChannels/ringBufferBases grouped by the VDIF thread
    // that carries them, so a frame only decodes the slots it actually holds.
    std::vector<std::vector<uint32_t>> slotsPerThread;

    MultiArrayHostBuffer<char, 4> *hostRingBuffer;
    SparseSet<TimeStamp>	validData;
    TimeStamp			latestWriteTime;
    std::mutex			validDataMutex, latestWriteTimeMutex;
    std::atomic<bool>		stop;

    SynchronizedReaderAndWriter readerAndWriterSynchronization;
    std::thread			inputThread, logThread;
    std::unique_ptr<std::thread> noInputThreadPtr;

    static volatile std::sig_atomic_t signalCaught;
};



#endif //RADIOBLOCKS_INPUTBUFFER_H

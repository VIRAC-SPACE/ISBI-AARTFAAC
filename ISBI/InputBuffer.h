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
    InputBuffer(const ISBI_Parset &, MultiArrayHostBuffer<char, 4> hostRingBuffer[], unsigned myFirstSubband, unsigned myNrSubbands, unsigned myFirstStation, unsigned myNrStations, unsigned nrTimesPerPacket);
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

    void handleConsecutivePackets(std::array<std::array<char, maxPacketSize>, maxNrPacketsInBuffer>& packetBuffer, unsigned firstPacket, unsigned lastPacket);
    SparseSet<TimeStamp> getCurrentValidData(const TimeStamp &earlyStartTime, const TimeStamp &endTime);

    const ISBI_Parset	&ps;
    unsigned			myFirstSubband, myNrSubbands, myFirstStation, myNrStations, nrRingBufferSamplesPerSubband, nrTimesPerPacket;
    std::vector<uint32_t>	mappedChannels;
    std::vector<int8_t *>	ringBufferBases;

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

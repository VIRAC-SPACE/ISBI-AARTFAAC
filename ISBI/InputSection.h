#ifndef INPUT_SECTION_H
#define INPUT_SECTION_H

#include "ISBI/Parset.h"
#include "ISBI/InputBuffer.h"
#include "Common/CUDA_Support.h"
#include "Common/PerformanceCounter.h"
#include "Common/SparseSet.h"
#include "Common/TimeStamp.h"

#include <vector>


class InputSection
{
  public:
    InputSection(const ISBI_Parset &);
    InputSection();
    ~InputSection();
    
    void fillInMissingSamples(const TimeStamp &, unsigned subband, std::vector<SparseSet<TimeStamp> > &validData);
    void enqueueHostToDeviceCopy(cu::Stream &, cu::DeviceMemory &devBuffer, PerformanceCounter &, const TimeStamp &, unsigned subband, std::vector<int64_t> integerStationDelays);

    void startReadTransaction(const TimeStamp &);
    void endReadTransaction(const TimeStamp &);

  private:
    const ISBI_Parset &ps;
  
  public:
    std::vector<MultiArrayHostBuffer<char, 4>> hostRingBuffers;

  private:
    std::vector<std::unique_ptr<InputBuffer>> inputBuffers;
    unsigned nrRingBufferSamplesPerSubband;
    const static unsigned nrTimesPerPacket = 2000;
};

#endif


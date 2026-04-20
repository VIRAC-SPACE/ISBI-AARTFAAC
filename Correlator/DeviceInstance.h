#if !defined DEVICE_INSTANCE_H
#define DEVICE_INSTANCE_H

#include "Common/Affinity.h"
#include "Common/CUDA_Support.h"
#include "libfilter/Filter.h"
#include "Correlator/TCC.h"

#include <functional>
#include <future>
#include <memory>
#include <mutex>

#define NR_DEV_VISIBILITIES_BUFFERS	2


class CorrelatorPipeline;

class DeviceInstance
{
  public:
    DeviceInstance(CorrelatorPipeline &, unsigned deviceNr);
    ~DeviceInstance();

    virtual void doSubband(const TimeStamp &,
		   unsigned subband,
		   std::function<void (cu::Stream &, cu::DeviceMemory &devInputBuffer, PerformanceCounter &)> &enqueueHostToDeviceTransfer,
		   const MultiArrayHostBuffer<char, 4> &hostInputBuffer,
		   const MultiArrayHostBuffer<float, 3> &hostDelaysAtBegin,
		   const MultiArrayHostBuffer<float, 3> &hostDelaysAfterEnd,
		   MultiArrayHostBuffer<std::complex<int32_t>, 4> &hostVisibilities,
		   unsigned startIndex = 0
		  );

    void doSubband(const TimeStamp &,
		   unsigned subband,
		   const MultiArrayHostBuffer<char, 4> &hostInputBuffer,
		   const MultiArrayHostBuffer<float, 3> &hostDelaysAtBegin,
		   const MultiArrayHostBuffer<float, 3> &hostDelaysAfterEnd,
		   MultiArrayHostBuffer<std::complex<int32_t>, 4> &hostVisibilities
		  );

    CorrelatorPipeline		&pipeline;
    const CorrelatorParset	&ps;

#if defined CL_DEVICE_TOPOLOGY_AMD
    bool			supportNumaAMD;
    unsigned			numaNode;
    std::unique_ptr<BoundThread> boundThread;
#endif

  public:
    cu::Device			device;
    cu::Context			context;
    cu::Stream			executeStream;

  protected:
    std::future<tcc::Filter>		filterFuture, filterOddFuture; // compile asynchronously
    std::future<TCC>		tccFuture; // compile asynchronously
    cu::DeviceMemory		devCorrectedData;

    tcc::Filter			filter;
    tcc::Filter			filterOdd;
    TCC				tcc;

    std::mutex			enqueueMutex;
    TimeStamp			previousTime;

#if defined CL_DEVICE_TOPOLOGY_AMD
    unsigned			lockNumber;
#endif
};


class DeviceInstanceWithoutUnifiedMemory : public DeviceInstance
{
  public:
    DeviceInstanceWithoutUnifiedMemory(CorrelatorPipeline &, unsigned deviceNr);

    virtual void doSubband(const TimeStamp &,
		   unsigned subband,
		   std::function<void (cu::Stream &, cu::DeviceMemory &devInputBuffer, PerformanceCounter &)> &enqueueHostToDeviceTransfer,
		   const MultiArrayHostBuffer<char, 4> &hostInputBuffer,
		   const MultiArrayHostBuffer<float, 3> &hostDelaysAtBegin,
		   const MultiArrayHostBuffer<float, 3> &hostDelaysAfterEnd,
		   MultiArrayHostBuffer<std::complex<int32_t>, 4> &hostVisibilities,
		   unsigned startIndex = 0
		  );

    cu::Stream			  hostToDeviceStream, deviceToHostStream;
    cu::DeviceMemory              devInputBuffer;
    cu::DeviceMemory              devFracDelays;
    cu::DeviceMemory		  devDelaysAtBegin, devDelaysAfterEnd;
    std::vector<cu::DeviceMemory> devVisibilities;//[NR_DEV_VISIBILITIES_BUFFERS];
    unsigned			  currentVisibilityBuffer;
    cu::Event			  inputDataFree, visibilityDataFree[NR_DEV_VISIBILITIES_BUFFERS];
};

#endif

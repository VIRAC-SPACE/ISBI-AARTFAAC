#if !defined DEVICE_INSTANCE_H
#define DEVICE_INSTANCE_H

#include "Common/Affinity.h"
#include "Common/CUDA_Support.h"
#include "Correlator/Filter.h"
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
		   const MultiArrayHostBuffer<float, 2> &hostDelays,
		   MultiArrayHostBuffer<std::complex<float>, 4> &hostVisibilities,
		   unsigned startIndex = 0
		  );

    void doSubband(const TimeStamp &,
		   unsigned subband,
		   const MultiArrayHostBuffer<char, 4> &hostInputBuffer,
		   const MultiArrayHostBuffer<float, 2> &hostDelays,
		   MultiArrayHostBuffer<std::complex<float>, 4> &hostVisibilities
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
    std::array<std::future<std::unique_ptr<Filter>>, 2> filterFutures;
    std::future<TCC>		tccFuture; // compile asynchronously
    cu::DeviceMemory		devCorrectedData;
    cu::DeviceMemory    devInputBuffer;

    std::array<std::unique_ptr<Filter>, 2> filters;
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
		   const MultiArrayHostBuffer<float, 2> &hostDelays,
		   MultiArrayHostBuffer<std::complex<float>, 4> &hostVisibilities,
		   unsigned startIndex = 0
		  );

    cu::Stream			  hostToDeviceStream, deviceToHostStream;
    cu::DeviceMemory		  devDelays;
    std::vector<cu::DeviceMemory> devVisibilities;//[NR_DEV_VISIBILITIES_BUFFERS];
    unsigned			  currentVisibilityBuffer;
    cu::Event			  inputDataFree, visibilityDataFree[NR_DEV_VISIBILITIES_BUFFERS];
};

#endif

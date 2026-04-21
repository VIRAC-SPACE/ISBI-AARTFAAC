#include "Common/Config.h"
#include "Common/PowerSensor.h"
#include "Correlator/CorrelatorPipeline.h"
#include "Correlator/DeviceInstance.h"

#include <cudawrappers/nvrtc.hpp>

#include <cuda_fp16.h>
#include <omp.h>

#if 0 && defined CL_DEVICE_TOPOLOGY_AMD
inline static cpu_set_t cpu_and(const cpu_set_t &a, const cpu_set_t &b)
{
  cpu_set_t c;
  CPU_AND(&c, &a, &b);
  return c;
}
#endif

namespace {
inline unsigned filterIndexForSubband(unsigned subband)
{
  return (subband & 1) == 0 ? 1U : 0U;
}
}


extern const char _binary_Correlator_Kernels_Transpose_cu_start, _binary_Correlator_Kernels_Transpose_cu_end;

DeviceInstance::DeviceInstance(CorrelatorPipeline &pipeline, unsigned deviceNr)
:
  pipeline(pipeline),
  ps(pipeline.ps),

#if 0 && defined CL_DEVICE_TOPOLOGY_AMD
  supportNumaAMD(device.getInfo<CL_DEVICE_VENDOR_ID>() == 0x1002),
  numaNode(supportNumaAMD ? getNUMAnodeOfPCIeDevice(device.getInfo<CL_DEVICE_TOPOLOGY_AMD>().pcie.bus & 255,
						    device.getInfo<CL_DEVICE_TOPOLOGY_AMD>().pcie.device & 255,
						    device.getInfo<CL_DEVICE_TOPOLOGY_AMD>().pcie.function & 255) : 0),
  boundThread(supportNumaAMD ? new BoundThread(ps.allowedCPUs(numaNode)) : nullptr),
//#else
  //numaNode(0),
#endif

  device(deviceNr),
  context(CU_CTX_SCHED_BLOCKING_SYNC, device),
  //integratedMemory(device.getAttribute(CU_DEVICE_ATTRIBUTE_INTEGRATED)),

  filterFutures {{
    std::async([&] {
      context.setCurrent();
      return std::make_unique<Filter>(device, ps, false);
    }),
    std::async([&] {
      context.setCurrent();
      return std::make_unique<Filter>(device, ps, true);
    }),
  }},

  tccFuture(std::async([&] {
    context.setCurrent();
    return TCC(device, ps);
  })),

  devCorrectedData((size_t) ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * ps.nrStations() * ps.nrPolarizations() * sizeof(__half2))),
  devInputBuffer((size_t) ps.nrStations() * ps.nrPolarizations() * (ps.nrSamplesPerChannel() + NR_TAPS - 1) * ps.nrChannelsPerSubbandBeforeFilter() * ps.nrBytesPerRealSample()),
  
  filters([&] {
    std::array<std::unique_ptr<Filter>, 2> result;
    result[0] = filterFutures[0].get();
    result[1] = filterFutures[1].get();
    return result;
  } ()),

  tcc(tccFuture.get()),

  previousTime(~0)
{
#if 0 && defined CL_DEVICE_TOPOLOGY_AMD
  if (supportNumaAMD) {
    unsigned bus = device.getInfo<CL_DEVICE_TOPOLOGY_AMD>().pcie.bus & 255;

#pragma omp critical (clog)
    std::clog << "GPU on bus " << bus << " is on NUMA node " << numaNode << std::endl;

    switch (bus) {
      case 0x06:
      case 0x07:  lockNumber = 0;
		  break;

      case 0x0D:
      case 0x0E:
      case 0x11:
      case 0x12:  lockNumber = 1;
		  break;

      case 0x85:
      case 0x86:
      case 0x87:
      case 0x88:
      case 0x89:
      case 0x8A:	
      case 0x8B:  lockNumber = 2;
		  break;

      default:	  std::clog << "Warning: device at unknown bus number " << bus << std::endl;
    		  lockNumber = 3;
    }
  }
#endif

  // do not wait for these events during first iteration, so generate them
  // already once during initialization
  //executeStream.record(inputDataFree);

  executeStream.synchronize();

#if defined CL_DEVICE_TOPOLOGY_AMD
  boundThread = nullptr;
#endif
}


DeviceInstanceWithoutUnifiedMemory::DeviceInstanceWithoutUnifiedMemory(CorrelatorPipeline &pipeline, unsigned deviceNr)
:
  DeviceInstance(pipeline, deviceNr),
  devDelays(ps.nrStations() * 2 * sizeof(float)),
  currentVisibilityBuffer(0)
{
  for (unsigned buffer = 0; buffer < NR_DEV_VISIBILITIES_BUFFERS; buffer ++)
    deviceToHostStream.record(visibilityDataFree[buffer]);

  for (unsigned i = 0; i < NR_DEV_VISIBILITIES_BUFFERS; i ++)
    devVisibilities.emplace_back(cu::DeviceMemory((size_t) ps.nrOutputChannelsPerSubband() * ps.nrBaselines() * ps.nrVisibilityPolarizations() * sizeof(std::complex<float>)));
}


DeviceInstance::~DeviceInstance()
{
  context.setCurrent();
}


#if 0 && defined CL_DEVICE_TOPOLOGY_AMD
// avoid that multiple GPUs share PCIe bandwidth

static std::mutex hostToDeviceLock[4], deviceToHostLock[4];

static void lockPCIeBus(cl_event event, cl_int status, void *arg)
{
  static_cast<std::mutex *>(arg)->lock();
}


static void unlockPCIeBus(cl_event event, cl_int status, void *arg)
{
  static_cast<std::mutex *>(arg)->unlock();
}

#endif


void DeviceInstance::doSubband(const TimeStamp &time,
			       unsigned subband,
			       std::function<void (cu::Stream &, cu::DeviceMemory &devInputBuffer, PerformanceCounter &)> &enqueueHostToDeviceTransfer,
			       const MultiArrayHostBuffer<char, 4> &hostInputBuffer,
			       const MultiArrayHostBuffer<float, 2> &hostDelays,
			       MultiArrayHostBuffer<std::complex<float>, 4> &hostVisibilities,
			       unsigned startIndex
			      )
{
  context.setCurrent();

  {
    std::lock_guard<std::mutex> lock(enqueueMutex);

    // filter.launchAsync(executeStream,
    //     	         devCorrectedData,
    //     		 cu::DeviceMemory(hostInputBuffer));

    cu::DeviceMemory devVisibilities(hostVisibilities);
    cu::DeviceMemory devCorrectedDataChannel0skipped(static_cast<CUdeviceptr>(devCorrectedData) + ps.nrSamplesPerChannel() * ps.nrStations() * ps.nrPolarizations() * sizeof(__half2));
    tcc.launchAsync(executeStream, devVisibilities, devCorrectedDataChannel0skipped, pipeline.correlateCounter);
  }

  executeStream.synchronize();
}


void DeviceInstanceWithoutUnifiedMemory::doSubband(const TimeStamp &time,
                                                   unsigned subband,
				                   std::function<void (cu::Stream &, cu::DeviceMemory &devInputBuffer, PerformanceCounter &)> &enqueueHostToDeviceTransfer,
				                   const MultiArrayHostBuffer<char, 4> &hostInputBuffer,
				                   const MultiArrayHostBuffer<float, 2> &hostDelays,
				                   MultiArrayHostBuffer<std::complex<float>, 4> &hostVisibilities,
				                   unsigned startIndex) {
  context.setCurrent();

  cu::Event inputTransferReady, computeReady, visibilityTransferReady;

  {
    std::lock_guard<std::mutex> lock(enqueueMutex);

    hostToDeviceStream.wait(inputDataFree);

    if (time != previousTime) {
      previousTime = time;

      hostToDeviceStream.memcpyHtoDAsync(devDelays, hostDelays.origin(), hostDelays.bytesize());
    }

    enqueueHostToDeviceTransfer(hostToDeviceStream, devInputBuffer, pipeline.samplesCounter);
    hostToDeviceStream.record(inputTransferReady);

#if 0 && defined CL_DEVICE_TOPOLOGY_AMD
    if (supportNumaAMD) {
      inputTransferStarted[0].setCallback(CL_SUBMITTED, &lockPCIeBus, &hostToDeviceLock[lockNumber]);
      inputTransferReady[0].setCallback(CL_RUNNING, &unlockPCIeBus, &hostToDeviceLock[lockNumber]);
    }
#endif

    //pipeline.samplesCounter.doOperation(inputTransferReady[0], 0, 0, bytesSent);

    executeStream.wait(inputTransferReady);

    std::optional<const cu::DeviceMemory> devFractionalDelays = ps.delayCompensation() ? std::optional<const cu::DeviceMemory>(devDelays) : std::nullopt;
    std::optional<double> subbandCenterFrequency = ps.delayCompensation() ? std::optional<double>(ps.subbandFrequencies()[subband]) : std::nullopt;

    filters[filterIndexForSubband(subband)]->launchAsync(executeStream,
                                                         devCorrectedData,
                                                         devInputBuffer,
                                                         pipeline.filterAndCorrectCounter,
                                                         devFractionalDelays,
                                                         subbandCenterFrequency);

    executeStream.record(inputDataFree);
    executeStream.wait(visibilityDataFree[currentVisibilityBuffer]);

    cu::DeviceMemory devCorrectedDataChannel0skipped(static_cast<CUdeviceptr>(devCorrectedData) + ps.nrSamplesPerChannel() * ps.nrStations() * ps.nrPolarizations() * sizeof(__half2));
    tcc.launchAsync(executeStream, devVisibilities[currentVisibilityBuffer], devCorrectedDataChannel0skipped, pipeline.correlateCounter);

    executeStream.record(computeReady);
    deviceToHostStream.wait(computeReady);

    {
      PerformanceCounter::Measurement measurement(pipeline.visibilitiesCounter, deviceToHostStream, 0, hostVisibilities.bytesize(), 0);
      deviceToHostStream.memcpyDtoHAsync(hostVisibilities.origin(), devVisibilities[currentVisibilityBuffer], hostVisibilities.bytesize());
    }

    deviceToHostStream.record(visibilityTransferReady);

#if 0 && defined CL_DEVICE_TOPOLOGY_AMD
    if (supportNumaAMD) {
      visibilityTransferReady.setCallback(CL_SUBMITTED, &lockPCIeBus, &deviceToHostLock[lockNumber]);
      visibilityTransferReady.setCallback(CL_RUNNING, &unlockPCIeBus, &deviceToHostLock[lockNumber]);
    }
#endif

    deviceToHostStream.record(visibilityDataFree[currentVisibilityBuffer]);
  }

  if (++ currentVisibilityBuffer == NR_DEV_VISIBILITIES_BUFFERS)
    currentVisibilityBuffer = 0;

  visibilityTransferReady.synchronize();
}


void DeviceInstance::doSubband(const TimeStamp &time,
			       unsigned subband,
			       const MultiArrayHostBuffer<char, 4> &hostInputBuffer,
			       const MultiArrayHostBuffer<float, 2> &hostDelays,
			       MultiArrayHostBuffer<std::complex<float>, 4> &hostVisibilities
			      )
{
  std::function<void (cu::Stream &, cu::DeviceMemory &, PerformanceCounter &)> enqueueHostToDeviceTransfer = [&] (cu::Stream &stream, cu::DeviceMemory &devInputBuffer, PerformanceCounter &counter) {
    PerformanceCounter::Measurement measurement(counter, stream, 0, 0, hostInputBuffer.bytesize());
    stream.memcpyHtoDAsync(devInputBuffer, hostInputBuffer, hostInputBuffer.bytesize());
  };

  doSubband(time, subband, enqueueHostToDeviceTransfer, hostInputBuffer, hostDelays, hostVisibilities);
}

#include "Common/Config.h"
#include "Common/PowerSensor.h"
#include "Correlator/CorrelatorPipeline.h"
#include "Correlator/DeviceInstance.h"

#include <cudawrappers/nvrtc.hpp>

#include <cuda_fp16.h>
#include <omp.h>

#include <iostream>
#include <cmath>

#if 0 && defined CL_DEVICE_TOPOLOGY_AMD
inline static cpu_set_t cpu_and(const cpu_set_t &a, const cpu_set_t &b)
{
  cpu_set_t c;
  CPU_AND(&c, &a, &b);
  return c;
}
#endif

#include <map>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace {
  auto getDelayAt (const std::map<int64_t, double>& delays, int64_t timestamp) {
    if (delays.empty()) {
      throw std::runtime_error("Delay map is empty");
    }

    // Exact match
    auto exact = delays.find(timestamp);
    if (exact != delays.end()) {
      return exact->second;
    }

    // Need at least 2 points to interpolate at all
    if (delays.size() < 2) {
      throw std::runtime_error("Not enough points to interpolate");
    }

    // Out of range check
    if (timestamp < delays.begin()->first || timestamp > delays.rbegin()->first) {
      throw std::runtime_error("Timestamp outside interpolation range");
    }

    // Flatten map into vectors for indexed access
    std::vector<double> x;
    std::vector<double> y;
    x.reserve(delays.size());
    y.reserve(delays.size());

    for (const auto& [ts, val] : delays) {
      x.push_back(static_cast<double>(ts));
      y.push_back(val);
    }

    const int n = static_cast<int>(x.size());

    // Find interval [x[i], x[i+1]] containing timestamp
    auto upper = std::upper_bound(x.begin(), x.end(), static_cast<double>(timestamp));
    int i = static_cast<int>(std::distance(x.begin(), upper)) - 1;

    // Safety clamp
    i = std::max(0, std::min(i, n - 2));

    const double xt = static_cast<double>(timestamp);

    // Fallback to linear interpolation if not enough points for Akima
    if (n < 5) {
      double t = (xt - x[i]) / (x[i + 1] - x[i]);
      return y[i] + t * (y[i + 1] - y[i]);
    }

    // Slopes between consecutive points
    std::vector<double> m(n - 1);
    for (int j = 0; j < n - 1; ++j) {
      m[j] = (y[j + 1] - y[j]) / (x[j + 1] - x[j]);
    }

    // Derivatives at each point
    std::vector<double> d(n);

    auto akimaDerivative = [&](int k) -> double {
      // For edges, fall back to simpler estimates
      if (k < 2 || k > n - 3) {
        if (k == 0) return m[0];
        if (k == 1) return 0.5 * (m[0] + m[1]);
        if (k == n - 2) return 0.5 * (m[n - 3] + m[n - 2]);
        if (k == n - 1) return m[n - 2];
      }

      double w1 = std::abs(m[k + 1] - m[k]);
      double w2 = std::abs(m[k - 1] - m[k - 2]);

      if (w1 + w2 == 0.0) {
        return 0.5 * (m[k - 1] + m[k]);
      }

      return (w1 * m[k - 1] + w2 * m[k]) / (w1 + w2);
    };

    for (int k = 0; k < n; ++k) {
      d[k] = akimaDerivative(k);
    }

    // Cubic Hermite interpolation on interval [i, i+1]
    double h = x[i + 1] - x[i];
    double t = (xt - x[i]) / h;

    double h00 =  2.0 * t * t * t - 3.0 * t * t + 1.0;
    double h10 =        t * t * t - 2.0 * t * t + t;
    double h01 = -2.0 * t * t * t + 3.0 * t * t;
    double h11 =        t * t * t -       t * t;

    return h00 * y[i]
      + h10 * h * d[i]
      + h01 * y[i + 1]
      + h11 * h * d[i + 1];
  };
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

  filterFuture(std::async([&] {
    context.setCurrent();
     
    tcc::FilterArgs filterArgs;
    filterArgs.nrReceivers = ps.nrStations();
    filterArgs.nrChannels = ps.nrChannelsPerSubband();
    filterArgs.nrSamplesPerChannel = ps.nrSamplesPerChannel();
    filterArgs.nrPolarizations = ps.nrPolarizations();

    filterArgs.input = tcc::FilterArgs::Input {
	.sampleFormat = tcc::FilterArgs::Format::i8,
	.isPurelyReal = true
    };

    filterArgs.firFilter = tcc::FilterArgs::FIR_Filter {
    	.nrTaps = 16,
        .sampleFormat = tcc::FilterArgs::Format::fp32
    };

    filterArgs.fft = tcc::FilterArgs::FFT {
    	.sampleFormat = tcc::FilterArgs::Format::fp32,
        .shift = false,
        .mirror = false 
    };

    filterArgs.delays = {
      .subbandBandwidth = ps.subbandBandwidth(),
      .polynomialOrder = 1,
      .separatePerPolarization = false
    };
    
    filterArgs.output = tcc::FilterArgs::Output {
      .sampleFormat = tcc::FilterArgs::Format::i8,
      .scaleFactor = std::nullopt
    };


    return tcc::Filter(device, filterArgs);
  })),

  filterOddFuture(std::async([&] {
    context.setCurrent();
     
    tcc::FilterArgs filterArgs;
    filterArgs.nrReceivers = ps.nrStations();
    filterArgs.nrChannels = ps.nrChannelsPerSubband();
    filterArgs.nrSamplesPerChannel = ps.nrSamplesPerChannel();
    filterArgs.nrPolarizations = ps.nrPolarizations();

    filterArgs.input = tcc::FilterArgs::Input {
	.sampleFormat = tcc::FilterArgs::Format::i8,
	.isPurelyReal = true
    };

    filterArgs.firFilter = tcc::FilterArgs::FIR_Filter {
    	.nrTaps = 16,
        .sampleFormat = tcc::FilterArgs::Format::fp32
    };

    filterArgs.fft = tcc::FilterArgs::FFT {
    	.sampleFormat = tcc::FilterArgs::Format::fp32,
        .shift = false,
        .mirror = true 
    };

    filterArgs.delays = {
      .subbandBandwidth = ps.subbandBandwidth(),
      .polynomialOrder = 1,
      .separatePerPolarization = false
    };
    
    filterArgs.output = tcc::FilterArgs::Output {
      .sampleFormat = tcc::FilterArgs::Format::i8,
      .scaleFactor = std::nullopt
    };


    return tcc::Filter(device, filterArgs);
  })),

  tccFuture(std::async([&] {
    context.setCurrent();
    return TCC(device, ps);
  })),

  devCorrectedData((size_t) ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * ps.nrStations() * ps.nrPolarizations() * ps.nrBytesPerComplexSample()),
  
  filter(filterFuture.get()),
  filterOdd(filterOddFuture.get()),
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
  devInputBuffer((size_t) ps.nrStations() * ps.nrPolarizations() * (ps.nrSamplesPerChannel() + NR_TAPS - 1) * ps.nrChannelsPerSubbandBeforeFilter() * ps.nrBytesPerRealSample()),
  devDelaysAtBegin(ps.nrBeams() * ps.nrStations() * ps.nrPolarizations() * sizeof(float)),
  devDelaysAfterEnd(ps.nrBeams() * ps.nrStations() * ps.nrPolarizations() * sizeof(float)),
  devFracDelays(sizeof(float) * ps.nrStations() * 2),
  currentVisibilityBuffer(0)
{
  for (unsigned buffer = 0; buffer < NR_DEV_VISIBILITIES_BUFFERS; buffer ++)
    deviceToHostStream.record(visibilityDataFree[buffer]);

  for (unsigned i = 0; i < NR_DEV_VISIBILITIES_BUFFERS; i ++)
    devVisibilities.emplace_back(cu::DeviceMemory((size_t) ps.nrOutputChannelsPerSubband() * ps.nrBaselines() * ps.nrVisibilityPolarizations() * sizeof(std::complex<int32_t>)));
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
			       const MultiArrayHostBuffer<float, 3> &hostDelaysAtBegin,
			       const MultiArrayHostBuffer<float, 3> &hostDelaysAfterEnd,
			       MultiArrayHostBuffer<std::complex<int32_t>, 4> &hostVisibilities,
			       unsigned startIndex
			      )
{
  context.setCurrent();

  {
    std::lock_guard<std::mutex> lock(enqueueMutex);

    std::cout << "DeviceInstance:doSubband\n";
    filter.launchAsync(executeStream,
		         devCorrectedData,
			 cu::DeviceMemory(hostInputBuffer));

    cu::DeviceMemory devVisibilities(hostVisibilities);
    cu::DeviceMemory devCorrectedDataChannel0skipped(static_cast<CUdeviceptr>(devCorrectedData) + ps.nrSamplesPerChannel() * ps.nrStations() * ps.nrPolarizations() * ps.nrBytesPerComplexSample());
    tcc.launchAsync(executeStream, devVisibilities, devCorrectedDataChannel0skipped, pipeline.correlateCounter);
  }

  executeStream.synchronize();
}


void DeviceInstanceWithoutUnifiedMemory::doSubband(const TimeStamp &time,
                                                   unsigned subband,
				                   std::function<void (cu::Stream &, cu::DeviceMemory &devInputBuffer, PerformanceCounter &)> &enqueueHostToDeviceTransfer,
				                   const MultiArrayHostBuffer<char, 4> &hostInputBuffer,
				                   const MultiArrayHostBuffer<float, 3> &hostDelaysAtBegin,
				                   const MultiArrayHostBuffer<float, 3> &hostDelaysAfterEnd,
				                   MultiArrayHostBuffer<std::complex<int32_t>, 4> &hostVisibilities,
				                   unsigned startIndex) {
  context.setCurrent();

  cu::Event inputTransferReady, computeReady, visibilityTransferReady;

  {
    std::lock_guard<std::mutex> lock(enqueueMutex);

    hostToDeviceStream.wait(inputDataFree);

    int referenceStation = 0;
    const double Fs = (double)ps.sampleRate();
    const double N  = (double)ps.nrSamplesPerChannel();

    float hostDelays[ps.nrStations()][2];

    for (std::size_t station = 0; station < ps.nrStations(); ++station) {
      double delayAtStart  = getDelayAt(ps.delays()[station], (int64_t)time);
      double delayAtEnd    = getDelayAt(ps.delays()[station], (int64_t)time + ps.nrSamplesPerSubbandBeforeFilter());


      if (station != referenceStation) {
        double delayAtStartR = getDelayAt(ps.delays()[referenceStation], (int64_t)time);
        double delayAtEndR   = getDelayAt(ps.delays()[referenceStation], (int64_t)time + ps.nrSamplesPerSubbandBeforeFilter());
        delayAtStart -= delayAtStartR;
        delayAtEnd -= delayAtEndR;
      } else {
        delayAtStart = 0.0;
        delayAtEnd = 0.0;
      }

      double delayInSamplesAtStart = delayAtStart * Fs;
      double delayInSamplesAtEnd = delayAtEnd * Fs;

      int integerDelayAtStart = static_cast<int>(std::llround(delayInSamplesAtStart));
      int integerDelayAtEnd = static_cast<int>(std::llround(delayInSamplesAtEnd));

      double fractionalDelayAtStart = delayInSamplesAtStart - (double)integerDelayAtStart;
      double fractionalDelayAtEnd = delayInSamplesAtEnd - (double)integerDelayAtEnd;

      double d0 = fractionalDelayAtStart / Fs;
      double d1 = (delayAtEnd - delayAtStart) / N;

      hostDelays[station][0] = -(float)d0;
      hostDelays[station][1] = -(float)d1;

      if (subband == 0) {
        std::cout << "station=" << station
          << " delayInSamplesAtStart=" << delayInSamplesAtStart 
          << " delayInSamplesAtEnd=" << delayInSamplesAtEnd
          << " integerDelayAtStart=" << integerDelayAtStart
          << " integerDelayAtEnd=" << integerDelayAtEnd
          << " fractionalDelayAtStart=" << fractionalDelayAtStart
          << " fractionalDelayAtEnd=" << fractionalDelayAtEnd
          << " d0=" << -float(d0)
          << " d1=" << -float(d1)
          << std::endl;
      }
    }

    hostToDeviceStream.memcpyHtoDAsync(devFracDelays, hostDelays, sizeof(float) * ps.nrStations() * 2);

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

    const double subbandCenter = ps.centerFrequencies()[subband];
    const bool mirrored = ((subband + 1) % 2) != 0;

    if (!mirrored) {
      filter.launchAsync(executeStream, devCorrectedData, devInputBuffer, devFracDelays, subbandCenter);
    } else {
      filterOdd.launchAsync(executeStream, devCorrectedData, devInputBuffer, devFracDelays, subbandCenter);
    }

    executeStream.record(inputDataFree);
    executeStream.wait(visibilityDataFree[currentVisibilityBuffer]);

    cu::DeviceMemory devCorrectedDataChannel0skipped(static_cast<CUdeviceptr>(devCorrectedData) + ps.nrSamplesPerChannel() * ps.nrStations() * ps.nrPolarizations() * ps.nrBytesPerComplexSample());
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
			       const MultiArrayHostBuffer<float, 3> &hostDelaysAtBegin,
			       const MultiArrayHostBuffer<float, 3> &hostDelaysAfterEnd,
			       MultiArrayHostBuffer<std::complex<float>, 4> &hostVisibilities
			      )
{
  std::function<void (cu::Stream &, cu::DeviceMemory &, PerformanceCounter &)> enqueueHostToDeviceTransfer = [&] (cu::Stream &stream, cu::DeviceMemory &devInputBuffer, PerformanceCounter &counter) {
    PerformanceCounter::Measurement measurement(counter, stream, 0, 0, hostInputBuffer.bytesize());
    stream.memcpyHtoDAsync(devInputBuffer, hostInputBuffer, hostInputBuffer.bytesize());
  };

  doSubband(time, subband, enqueueHostToDeviceTransfer, hostInputBuffer, hostDelaysAtBegin, hostDelaysAfterEnd, hostVisibilities);
}

ARCH=			$(shell arch)	
SRC_DIR=		\"$(shell pwd)\" # FIXME

BOOST_INCLUDE ?=	$(BOOST_ROOT)/include
BOOST_LIB ?=		$(BOOST_ROOT)/lib

CUDA_INCLUDE ?=		$(CUDA_ROOT)/include
ifeq ("$(ARCH)", "x86_64")
CUDA_LIB ?=		$(CUDA_ROOT)/targets/x86_64-linux/lib/stubs
endif
ifeq ("$(ARCH)", "aarch64")
CUDA_LIB ?=		$(CUDA_ROOT)/targets/sbsa-linux/lib/stubs
endif

ifneq ("$(POWER_SENSOR3_ROOT)", "")
POWER_SENSOR3_INCLUDE ?=$(POWER_SENSOR3_ROOT)/host/include -DMEASURE_POWER
POWER_SENSOR3_LIB ?=	$(POWER_SENSOR3_ROOT)/build-$(ARCH)/host
endif

NVRTC_INCLUDE ?=	$(CUDA_ROOT)/include
NVRTC_LIB ?=		$(CUDA_ROOT)/lib64

FFTW_INCLUDE ?=		$(FFTW_ROOT)/include
FFTW_LIB ?=		$(FFTW_ROOT)/lib #64

FILTER_INCLUDE=		$(FILTER_ROOT)
FILTER_LIB=		$(FILTER_ROOT)/build-x86_64/libfilter

TCC_INCLUDE=		$(TCC_ROOT)
TCC_LIB=		$(TCC_ROOT)/build/libtcc

CUDA_WRAPPERS_INCLUDE=	$(TCC_ROOT)/build/_deps/cudawrappers-src/include


CXX=			g++
CXXFLAGS=		-std=c++17\
			-DSRC_DIR=$(SRC_DIR)\
			-march=native\
			-g -O3\
			-fopenmp\
			-I.\
			-I$(CUDA_INCLUDE)\
			-I$(NVRTC_INCLUDE)\
			-I$(BOOST_INCLUDE)\
			-I$(FFTW_INCLUDE)\
			-I$(CUDA_WRAPPERS_INCLUDE)\
			-I$(TCC_INCLUDE)\
			-I$(FILTER_INCLUDE)

ifneq ("$(POWER_SENSOR3_INCLUDE)", "")
CXXFLAGS +=		-I$(POWER_SENSOR3_INCLUDE)
endif

COMMON_SOURCES=		\
			Common/Affinity.cc\
			Common/BandPass.cc\
			Common/Exceptions/AddressTranslator.cc\
			Common/Exceptions/Backtrace.cc\
			Common/Exceptions/Exception.cc\
			Common/Exceptions/SymbolTable.cc\
			Common/Function.cc\
			Common/HugePages.cc\
			Common/LockedRanges.cc\
			Common/Module.cc\
			Common/Parset.cc\
			Common/PerformanceCounter.cc\
			Common/PowerSensor.cc\
			Common/ReaderWriterSynchronization.cc\
			Common/SystemCallException.cc\
			Common/Stream/Descriptor.cc\
			Common/Stream/FileDescriptorBasedStream.cc\
			Common/Stream/FileStream.cc\
			Common/Stream/NamedPipeStream.cc\
			Common/Stream/NullStream.cc\
			Common/Stream/SharedMemoryStream.cc\
			Common/Stream/SocketStream.cc\
			Common/Stream/Stream.cc\
			Common/Stream/StringStream.cc\
			Common/TimeStamp.cc

CORRELATOR_SOURCES=	$(COMMON_SOURCES)\
			Correlator/Correlator.cc\
			Correlator/CorrelatorPipeline.cc\
			Correlator/DeviceInstance.cc\
			Correlator/Kernels/Transpose.cu\
			Correlator/Kernels/TransposeKernel.cc\
			Correlator/Parset.cc\
			Correlator/TCC.cc\
			Correlator/Filter.cc

ISBI_SOURCES =		$(COMMON_SOURCES)\
                        ISBI/isbi.cc\
			ISBI/VDIFStream.cc\
                        ISBI/CorrelatorPipeline.cc\
                        ISBI/CorrelatorWorkQueue.cc\
                        ISBI/InputBuffer.cc\
                        ISBI/InputSection.cc\
                        ISBI/OutputBuffer.cc\
                        ISBI/OutputSection.cc\
                        ISBI/Parset.cc\
                        ISBI/Visibilities.cc\
												ISBI/DelayCorrection.cc\
                        Correlator/CorrelatorPipeline.cc\
                        Correlator/Parset.cc\
                        Correlator/DeviceInstance.cc\
                        Correlator/Kernels/Transpose.cu\
                        Correlator/Kernels/TransposeKernel.cc\
                        Correlator/Parset.cc\
                        Correlator/TCC.cc\
												Correlator/Filter.cc


ALL_SOURCES=		$(sort\
			   $(CORRELATOR_SOURCES)\
			   $(CORRELATOR_DEVICE_INSTANCE_TEST_SOURCES)\
			   $(ISBI_SOURCES)\
			 )

CORRELATOR_OBJECTS=	$(patsubst %.cu,%.o,$(CORRELATOR_SOURCES:%.cc=%.o))
CORRELATOR_DEVICE_INSTANCE_TEST_OBJECTS=$(patsubst %.cu,%.o,$(CORRELATOR_DEVICE_INSTANCE_TEST_SOURCES:%.cc=%.o))
ISBI_OBJECTS=		$(patsubst %.cu,%.o,$(ISBI_SOURCES:%.cc=%.o))

ALL_OBJECTS=		$(patsubst %.cu,%.o,$(ALL_SOURCES:%.cc=%.o))
DEPENDENCIES=		$(patsubst %.cu,%.d,$(ALL_SOURCES:%.cc=%.d))

EXECUTABLES=            Correlator/Correlator\
			ISBI/ISBI

LIBRARIES+=		-L${BOOST_LIB} -lboost_program_options
LIBRARIES+=		-L${FFTW_LIB} -lfftw3f
LIBRARIES+=		-L${FILTER_LIB} -Wl,-rpath=$(FILTER_LIB) -lfilter
LIBRARIES+=		-L${TCC_LIB} -Wl,-rpath=$(TCC_LIB) -ltcc
LIBRARIES+=		-L${NVRTC_LIB} -Wl,-rpath=$(NVRTC_LIB) -lnvrtc
LIBRARIES+=             -L/var/software/spack-builtin/20250403/opt/spack/linux-rocky8-zen2/gcc-12.2.0/cuda-12.6.2-v5h46v5wx6zglkfx3zrq2oq6tk73bnbx/targets/x86_64-linux/lib/stubs -Wl,-rpath=/var/software/spack-builtin/20250403/opt/spack/linux-rocky8-zen2/gcc-12.2.0/cuda-12.6.2-v5h46v5wx6zglkfx3zrq2oq6tk73bnbx/targets/x86_64-linux/lib/stubs -lcuda
LIBRARIES+=		-lnuma

ifneq ("$(POWER_SENSOR3_LIB)", "")
LIBRARIES+=		-L$(POWER_SENSOR3_LIB) -lPowerSensor
endif


%.d:			%.cc
			-$(CXX) $(CXXFLAGS) -MM -MT $@ -MT ${@:%.d=%.o} $< -o $@

%.o:			%.cc
			$(CXX) $(CXXFLAGS) -o $@ -c $<

%.o:			%.cu # CUDA code embedded in object file
			ld -r -b binary -o $@ $<

%:			%.tar.gz
			tar xfz $<


all::			$(EXECUTABLES)

clean::
			rm -rf $(ALL_OBJECTS) $(DEPENDENCIES) $(EXECUTABLES) nvidia-mathdx-22.11.0-Linux.tar.gz nvidia-mathdx-22.11.0-Linux

Correlator/Correlator:	$(CORRELATOR_OBJECTS)
			$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBRARIES)

ISBI/ISBI:              $(ISBI_OBJECTS)
			$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBRARIES)

ifeq (0, $(words $(findstring $(MAKECMDGOALS), clean)))
-include $(DEPENDENCIES)
endif

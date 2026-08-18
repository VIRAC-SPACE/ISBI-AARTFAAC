# General
`git clone git@git.astron.nl:RD/AARTFAAC.git`

## Pre-requisites
On DAS-5 requires module files (`load module <>`): `gcc/8.3.0`, `cuda110/toolkit/11.0.1`, `boost/1.73-gcc-8.3.0` and `fftw/3.3.8-gcc-8.3.0`

For usage with Nvidia GPUs, use the CUDA OpenCL library instead of rocm, current as work around: `export OPENCL_LIB=$CUDA_LIB`

# Correlator

## Installation
```
cd AARTFAAC
make -j
```

## Usage
A test program for the Correlator pipeline `./Correlator/Correlator`

Unit tests in `/Correlator/Tests` :
* `CorrelatorTest`;
Uses 288 stations by default and therefore does not run subtest `CorrelateRectangleTest` (only used when nr of stations is not a multiple of 32)
* `DelayAndBandPassTest`
* `DeviceInstanceTest`
* `Filter_FFT_Test`
* `FIR_FilterTest`

# AARTFAAC

## Installation
```
cd AARTFAAC
mkdir AARTFAAC/installed
make install
```

## Usage
A test program `./AARTFAAC/installed/AARTFAAC`, currently fails (not intended for single node?)
# ISBI

## Input modes

`ISBI/ISBI` reads VDIF frames either from recordings or from the network. The
mode is selected with the real-time flag `-R` (`--realTime`), and the input
sources are given per station with `-i` (`--inputDescriptors`).

Offline (default, `-R false`): every input descriptor is the path of a VDIF
recording. Each station's reader seeks to
`startTime - historySamples - maxDelay` and then reads frames sequentially, so
the run ends when a recording runs out.

```
./ISBI/ISBI -R false -i /data/station0.vdif,/data/station1.vdif -D 2025-08-18_12:00:00 -r 20 ...
```

Real-time (`-R true`): every input descriptor is a UDP endpoint written as
`[udp:][address:]port`, and one complete VDIF frame is expected per datagram.
The address is the local address to bind to; omit it (or use `0.0.0.0`) to
listen on all interfaces. Each station needs its own port.

```
./ISBI/ISBI -R true -i udp:0.0.0.0:50000,udp:0.0.0.0:50001 -r 20 ...
```

In real-time mode the reader never blocks indefinitely: when no datagram
arrives within 100 ms it hands the frames it already has to the correlator,
which also keeps running (without data) through `noInputThreadBody()`.
Datagrams whose size does not match their VDIF frame length, whose header is
invalid, or whose frame layout differs from the first frame received are
dropped and counted; the totals are reported when the stream is closed.

Frames arrive faster than the input thread is scheduled, so the receive buffer
matters: the socket asks for 64 MB but the kernel clamps it to
`net.core.rmem_max` (the value actually granted is logged at start-up).

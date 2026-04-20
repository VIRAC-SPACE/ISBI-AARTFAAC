#include "ISBI/DelayCorrection.h"

#include <fstream>
#include <map>
#include <vector>
#include <cstdint>

std::vector<std::map<int64_t, double>> DelayCorrection::readDelayFile() const {
  std::ifstream delayFile(ps.delayFile(), std::ios::binary);
  if (!delayFile.is_open()) {
    throw std::runtime_error("Could not open delay file: " + ps.delayFile());
  }

  std::vector<std::map<int64_t, double>> delays;
  delays.reserve(ps.nrStations());

  for (unsigned station = 0; station < ps.nrStations(); ++station) {
    uint32_t n;
    delayFile.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));

    std::map<int64_t, double> stationDelays;

    for (unsigned i = 0; i < n; ++i) {
      int64_t ts;
      double delay;

      delayFile.read(reinterpret_cast<char*>(&ts), sizeof(int64_t));
      delayFile.read(reinterpret_cast<char*>(&delay), sizeof(double));

      stationDelays[ts] = delay;
    }

    delays.push_back(std::move(stationDelays));
  }

  return delays;
}

DelayCorrection::DelayCorrection(const ISBI_Parset &ps) :
  ps(ps),
  rawDelays(readDelayFile()),
  referenceStation(0) {}

double DelayCorrection::getDelayAt(const int64_t &timestamp, unsigned station) const {
  if (rawDelays[station].empty()) {
    throw std::runtime_error("Delay map is empty.");
  }

  auto upper = rawDelays[station].lower_bound(timestamp);

  if (upper != rawDelays[station].end() && upper->first == timestamp) {
    return upper->second;
  }

  if (upper == rawDelays[station].begin()) {
    throw std::runtime_error("Timestamp is before available data raneg.");
  }

  if (upper == rawDelays[station].end()) {
    throw std::runtime_error("Timestamp is after available data range.");
  }

  auto lower = std::prev(upper);

  int64_t t1 = lower->first;
  int64_t t2 = upper->first;
  double v1 = lower->second;
  double v2 = upper->second;

  double ratio = static_cast<double>(timestamp - t1) / (t2 - t1);

  return v1 + ratio * (v2 - v1);
}

std::vector<DelayCorrection::StationDelay> DelayCorrection::stationDelays(const TimeStamp &time) const {
  std::vector<DelayCorrection::StationDelay> result(ps.nrStations());

  TimeStamp endTime = time + ps.nrSamplesPerSubbandBeforeFilter();
  double Fs = (double)ps.sampleRate();
  double N = (double)ps.nrSamplesPerChannel();

  for (unsigned station = 0; station < ps.nrStations(); ++station) {
    double delayAtStart = getDelayAt(time, station);
    double delayAtEnd = getDelayAt(endTime, station);

    if (station != referenceStation) {
      double delayAtStartRef = getDelayAt(time, referenceStation);
      double delayAtEndRef = getDelayAt(time, referenceStation);

      delayAtStart -= delayAtStartRef;
      delayAtEnd -= delayAtEndRef;
    } else {
      delayAtStart = 0.0;
      delayAtEnd = 0.0;
    }

    double delaySamplesAtStart = delayAtStart * Fs;
    int64_t integerDelay = static_cast<int64_t>(std::llround(delaySamplesAtStart));

    double fractionalDelay = delaySamplesAtStart - (double)integerDelay;

    double d0 = fractionalDelay / Fs;
    double d1 = (delayAtEnd - delayAtStart) / N;

    result[station].integerSamples = integerDelay;
    result[station].d0 = -static_cast<float>(d0);
    result[station].d1 = -static_cast<float>(d1);
  }

  return result;
}



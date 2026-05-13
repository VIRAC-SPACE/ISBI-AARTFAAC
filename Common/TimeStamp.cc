#include "Common/Config.h"
#include "Common/TimeStamp.h"
#include "Common/SystemCallException.h"

#include <time.h>
#include <sys/time.h>

//#include <boost/date_time/posix_time/posix_time.hpp>


TimeStamp TimeStamp::fromDate(const char *date, unsigned clockSpeed)
{
#if 0
  boost::posix_time::ptime startTime = boost::posix_time::time_from_string(date);
  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  boost::posix_time::time_duration diff(startTime - epoch);
  double time = diff.total_seconds();
  return TimeStamp(time * clockSpeed / 1024, clockSpeed);
#else
  std::tm tm = {};
  strptime(date, "%Y-%m-%d %H:%M:%S", &tm);
  return TimeStamp((uint64_t) mktime(&tm) * clockSpeed, clockSpeed);
#endif
}


TimeStamp TimeStamp::now(unsigned clockSpeed)
{
  struct timeval tv;

  if (gettimeofday(&tv, nullptr) < 0)
    throw SystemCallException("gettimeofday", errno);

  double time = tv.tv_sec + tv.tv_usec * 1e-6;
  return TimeStamp(time * clockSpeed, clockSpeed);
}


void TimeStamp::wait() const
{
  signed seconds;

  while ((seconds = getSeqId() - now(clockSpeed).getSeqId()) > 0)
    sleep(seconds);
}


std::ostream &operator << (std::ostream &os, const TimeStamp &ts)
{
  return os << ts.getSeqId() << "s";
}

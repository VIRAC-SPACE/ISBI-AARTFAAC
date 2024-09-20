#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#define EVEN_SECOND_HAS_MORE_SAMPLES

#include <inttypes.h>

#include <chrono>
#include <iostream>

// FIXME: the 1024 constant factor in clockSpeed is a design flaw

class TimeStamp
{
  public:
    TimeStamp(); // empty constructor to be able to create vectors of TimeStamps
    TimeStamp(int64_t time); // for conversion from ints, used to convert values like 0x7FFFFFFF and 0x0 for special cases.
    TimeStamp(int64_t time, unsigned clockSpeed);
    TimeStamp(unsigned seqId, unsigned blockId, unsigned clockSpeed);

    static TimeStamp fromDate(const char *date, unsigned clockSpeed);
    static TimeStamp now(unsigned clockSpeed);
    void	     wait() const;

    TimeStamp	    &setStamp(unsigned seqId, unsigned blockId);
    unsigned	    getSeqId() const;
    unsigned	    getBlockId() const;
    unsigned	    getClock() const { return clockSpeed; }

    template <typename T> TimeStamp &operator += (T increment);
    template <typename T> TimeStamp &operator -= (T decrement);
			  TimeStamp  operator ++ (int); // postfix
			  TimeStamp &operator ++ ();	  // prefix
			  TimeStamp  operator -- (int);
			  TimeStamp &operator -- ();

    template <typename T> TimeStamp  operator +  (T) const;
    template <typename T> TimeStamp  operator -  (T) const;
			  int64_t    operator -  (const TimeStamp &) const;

			  bool       operator >  (const TimeStamp &) const;
			  bool       operator <  (const TimeStamp &) const;
			  bool       operator >= (const TimeStamp &) const;
			  bool       operator <= (const TimeStamp &) const;
			  bool       operator == (const TimeStamp &) const;
			  bool       operator != (const TimeStamp &) const;
			  bool       operator !  () const;

				     operator int64_t () const;
				     operator double () const;
				     operator struct timespec () const;
				     typedef std::chrono::time_point<std::chrono::system_clock,std::chrono::duration<double>> time_point;
				     operator time_point () const;

    friend std::ostream &operator << (std::ostream &os, const TimeStamp &ss);

  protected:
    int64_t  time;
    unsigned clockSpeed;
};

inline TimeStamp::TimeStamp()
:
  time(0),
  clockSpeed(0)
{
}

inline TimeStamp::TimeStamp(int64_t time)
:
  time(time),
  clockSpeed(0)
{
}

inline TimeStamp::TimeStamp(int64_t time, unsigned clockSpeed)
:
  time(time),
  clockSpeed(clockSpeed)
{
}

inline TimeStamp::TimeStamp(unsigned seqId, unsigned blockId, unsigned clockSpeed)
:
#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
  time(((int64_t) seqId * clockSpeed + 512) / 1024 + blockId),
#else
  time(((int64_t) seqId * clockSpeed) / 1024 + blockId),
#endif
  clockSpeed(clockSpeed)
{
}

inline TimeStamp &TimeStamp::setStamp(unsigned seqId, unsigned blockId)
{
#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
  time = ((int64_t) seqId * clockSpeed + 512) / 1024 + blockId;
#else
  time = ((int64_t) seqId * clockSpeed) / 1024 + blockId;
#endif
  return *this;
}

inline unsigned TimeStamp::getSeqId() const
{
#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
  return (unsigned) (1024 * time / clockSpeed);
#else
  return (unsigned) ((1024 * time + 512) / clockSpeed);
#endif
}

inline unsigned TimeStamp::getBlockId() const
{
#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
  return (unsigned) (1024 * time % clockSpeed / 1024);
#else
  return (unsigned) ((1024 * time + 512) % clockSpeed / 1024);
#endif
}

template <typename T> inline TimeStamp &TimeStamp::operator += (T increment)
{ 
  time += increment;
  return *this;
}

template <typename T> inline TimeStamp &TimeStamp::operator -= (T decrement)
{ 
  time -= decrement;
  return *this;
}

inline TimeStamp &TimeStamp::operator ++ ()
{ 
  ++ time;
  return *this;
}

inline TimeStamp TimeStamp::operator ++ (int)
{ 
  TimeStamp tmp = *this;
  ++ time;
  return tmp;
}

inline TimeStamp &TimeStamp::operator -- ()
{ 
  -- time;
  return *this;
}

inline TimeStamp TimeStamp::operator -- (int)
{ 
  TimeStamp tmp = *this;
  -- time;
  return tmp;
}

template <typename T> inline TimeStamp TimeStamp::operator + (T increment) const
{ 
  return TimeStamp(time + increment, clockSpeed);
}

template <typename T> inline TimeStamp TimeStamp::operator - (T decrement) const
{ 
  return TimeStamp(time - decrement, clockSpeed);
}

inline int64_t TimeStamp::operator - (const TimeStamp &other) const
{ 
  return time - other.time;
}

inline bool TimeStamp::operator ! () const
{
  return time == 0;
}

inline TimeStamp::operator int64_t () const
{
  return time;
}

inline TimeStamp::operator double () const
{
  return (double) 1024.0 * time / clockSpeed;
}

inline TimeStamp::operator struct timespec () const
{
  int64_t	  ns = (int64_t) (time * 1024 * 1e9 / clockSpeed);
  struct timespec ts;

  ts.tv_sec  = ns / 1000000000ULL;
  ts.tv_nsec = ns % 1000000000ULL;

  return ts;
}

inline TimeStamp::operator TimeStamp::time_point () const
{
  return time_point(std::chrono::duration<double>((double) *this));
}

inline bool TimeStamp::operator > (const TimeStamp &other) const
{ 
  return time > other.time;
}

inline bool TimeStamp::operator >= (const TimeStamp &other) const
{
  return time >= other.time;
}

inline bool TimeStamp::operator < (const TimeStamp &other) const
{ 
  return time < other.time;
}

inline bool TimeStamp::operator <= (const TimeStamp &other) const
{
  return time <= other.time;
}

inline bool TimeStamp::operator == (const TimeStamp &other) const
{ 
  return time == other.time;
}

inline bool TimeStamp::operator != (const TimeStamp &other) const
{ 
  return time != other.time;
}

#endif

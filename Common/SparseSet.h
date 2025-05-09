#ifndef COMMON_SPARSESET_H
#define COMMON_SPARSESET_H

#include <sys/types.h>
#include <stdint.h>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>


// A SparseSet<T> represents a sorted vector of [from, to) pairs of type
// T representing a boolean presence.

template <typename T> class SparseSet
{
  public:
    struct range {
      range()
      {
      }

      range(T begin, T end) : begin(begin), end(end)
      {
      }

      T begin, end;
    };

    typedef typename std::vector<range>     Ranges;
    typedef typename Ranges::iterator iterator;
    typedef typename Ranges::const_iterator const_iterator;

    // Empty SparseSet
    SparseSet();

    // Singleton SparseSet
    SparseSet(T index);
    
    // Single-range SparseSet
    SparseSet(T first /* inclusive */, T last /* exclusive */);

    // Add `index' to the set.
    SparseSet<T> &include(T index);

    // Add [first, last) to the set.
    SparseSet<T> &include(T first /* inclusive */, T last /* exclusive */);

    // Remove `index' from the set.
    SparseSet<T> &exclude(T index);

    // Remove [first, last) from the set.
    SparseSet<T> &exclude(T first /* inclusive */, T last /* exclusive */);

    // Clear the set.
    SparseSet<T> &reset();

    // Returns true iff the set is empty.
    bool	 empty() const;

    // Returns the number of elements in the set.
    T            count() const;

    // Returns true if `index' is in the set.
    bool         test(T index) const;

    // Compare two sets.
    bool operator == (const SparseSet<T> &) const;
    bool operator != (const SparseSet<T> &) const;

    // Return the union of two sets.
    SparseSet<T> operator | (const SparseSet<T> &) const;
    SparseSet<T> &operator |= (const SparseSet<T> &);

    // Return the intersection of two sets.
    SparseSet<T> operator & (const SparseSet<T> &) const;
    SparseSet<T> &operator &= (const SparseSet<T> &);

    // Increase all indices in the set by `count'.
    SparseSet<T> &operator += (size_t count);

    // Decrease all indices in the set by `count'.
    SparseSet<T> &operator -= (size_t count);

    // Divide all indices by `shrinkFactor'. Fractions are rounded in favor
    // of including more elements.
    SparseSet<T> &operator /= (size_t shrinkFactor);

    // Return the inverse of the set, considering the [first, last) range.
    // These parameters are needed because a SparseSet has no inherent
    // knowledge of the set's boundaries.
    SparseSet<T> invert(T first, T last) const;

    // Return the subset within [first, last).
    SparseSet<T> subset(T first, T last) const;

    // Returns the range vector, useful for iteration.
    const Ranges &getRanges() const;

    // Returns the number of bytes to marshall the given
    // number of ranges.
    static size_t marshallSize(size_t nrRanges);

    // Write the set to *ptr, using at most maxSize bytes.
    // Returns the number of marshalled bytes, or -1
    // if maxSize was too small.
    ssize_t marshall(void *ptr, size_t maxSize) const;

    // Read the SparseSet from *ptr.
    void    unmarshall(const void *ptr);

  private:
    Ranges ranges;

    struct less {
      bool operator() (const range &x, const range &y)
      {
	return x.end < y.begin;
      }
    };

    struct less_equal {
      bool operator() (const range &x, const range &y)
      {
	return x.end <= y.begin;
      }
    };
};

template <typename T>
std::ostream &operator << (std::ostream &str, const SparseSet<T> &set);


template <typename T>
inline SparseSet<T>::SparseSet()
{
}


template <typename T>
inline SparseSet<T>::SparseSet(T index)
{
  include(index);
}


template <typename T>
inline SparseSet<T>::SparseSet(T first, T last)
{
  include(first, last);
}


template <typename T>
inline SparseSet<T> &SparseSet<T>::include(T index)
{
  return include(index, index + 1);
}


template <typename T>
inline SparseSet<T> &SparseSet<T>::exclude(T index)
{
  return exclude(index, index + 1);
}


template <typename T>
inline SparseSet<T> &SparseSet<T>::reset()
{
  ranges.resize(0);
  return *this;
}


template <typename T>
inline SparseSet<T> &SparseSet<T>::operator |= (const SparseSet<T> &other)
{
  ranges = (*this | other).ranges;
  return *this;
}


template <typename T>
inline SparseSet<T> &SparseSet<T>::operator &= (const SparseSet<T> &other)
{
  ranges = (*this & other).ranges;
  return *this;
}


template <typename T>
inline SparseSet<T> SparseSet<T>::invert(T first, T last) const
{
  SparseSet<T> inverted;

  for (const_iterator it = ranges.begin(); it != ranges.end(); it++) {
    inverted.include(first, it->begin);
    first = it->end;
  }

  inverted.include(first, last);

  return inverted;
}


template <typename T>
inline SparseSet<T> SparseSet<T>::subset(T first, T last) const
{
  return SparseSet<T>(*this).exclude(last, ~0U).exclude(0, first);
}


//template <typename T> inline const std::vector<typename SparseSet<T>::range> &SparseSet<T>::getRanges() const
template <typename T>
inline const typename SparseSet<T>::Ranges & SparseSet<T>::getRanges() const
{
  return ranges;
}


template <typename T>
SparseSet<T> &SparseSet<T>::include(T first, T last)
{
  if (first < last) {
    // find two iterators that mark the first resp. last involved ranges
    range r(first, last);
    std::pair<iterator, iterator> iters = equal_range(ranges.begin(), ranges.end(), r, less());

    if (iters.first == iters.second) {
      // insert new tuple
      ranges.insert(iters.first, r);
    } else {
      // combine with existing tuple(s)
      iters.first->begin = std::min(first, iters.first->begin);
      iters.first->end = std::max(last, iters.second[-1].end);

      ranges.erase(iters.first + 1, iters.second);
    }
  }

  return *this;
}


template <typename T>
SparseSet<T> &SparseSet<T>::exclude(T first, T last)
{
  if (first < last) {
    // find two iterators that mark the first resp. last involved ranges
    // unlike in include(), a range that is adjacent to first or last is not
    // considered to be involved, hence the use of less_equal()
    std::pair<iterator, iterator> iters = equal_range(ranges.begin(), ranges.end(), range(first, last), less_equal());

    if (iters.first != iters.second) { // check if there are tuples involved
      if (iters.second - iters.first == 1 && first > iters.first->begin && last < iters.first->end) {
	// split tuple
	range r(last, iters.first->end);
	iters.first->end = first;
	ranges.insert(iters.second, r);
      } else {
	// possibly erase tuples
	if (first > iters.first->begin)
	  (iters.first++)->end = first;  // adjust first tuple; do not erase

	if (last < iters.second[-1].end)
	  (--iters.second)->begin = last;  // adjust last tuple; do not erase

	ranges.erase(iters.first, iters.second);
      }
    }
  }

  return *this;
}


template <typename T>
bool SparseSet<T>::empty() const
{
  return ranges.size() == 0;
}


template <typename T>
T SparseSet<T>::count() const
{
  T count = 0;

  for (const_iterator it = ranges.begin(); it != ranges.end(); it++)
    count += it->end - it->begin;

  return count;
}


template <typename T>
bool SparseSet<T>::test(T index) const
{
  const_iterator it = lower_bound(ranges.begin(), ranges.end(), range(index, index + 1), less_equal());
  return it != ranges.end() && index >= it->begin;
}


template <typename T>
bool SparseSet<T>::operator == (const SparseSet<T> &other) const
{
  const_iterator it1, it2;

  for (it1 = ranges.begin(), it2 = other.ranges.begin();
       it1 != ranges.end() && it2 != other.ranges.end();
       ++it1, ++it2) {
    if (it1->begin != it2->begin || it1->end != it2->end)
      return false;
  }

  return it1 == ranges.end() && it2 == other.ranges.end();
}


template <typename T>
bool SparseSet<T>::operator != (const SparseSet<T> &other) const
{
  return !(*this == other);
}


template <typename T>
SparseSet<T> SparseSet<T>::operator | (const SparseSet<T> &other) const
{
  // iterate with two iterators over both sets, comparing the ranges to decide
  // what to do: include a range from the first set, include a range from the
  // second set, or merge (multiple) ranges from both sets.

  SparseSet<T> union_set;
  const_iterator it1 = ranges.begin(), it2 = other.ranges.begin();

  while (it1 != ranges.end() && it2 != other.ranges.end()) {
    if (it1->end < it2->begin) {
      union_set.ranges.push_back(*it1++); // no overlap; *it1 is the smallest
    } else if (it2->end < it1->begin) {
      union_set.ranges.push_back(*it2++); // no overlap; *it2 is the smallest
    } else { // there is overlap, or it1 and it2 are contiguous
      T new_begin = std::min(it1->begin, it2->begin);

      // check if subsequent ranges from set1 and set2 must be joined as well
      while (1) {
	if (it1 + 1 != ranges.end() && it1[1].begin <= it2->end) {
	  ++it1;
	} else if (it2 + 1 != other.ranges.end() && it2[1].begin <= it1->end) {
	  ++it2;
	} else {
	  break;
	}
      }

      union_set.ranges.push_back(range(new_begin, std::max(it1->end, it2->end)));
      ++it1, ++it2;
    }
  }

  // possibly append the remainder of the set that we have not finished yet
  union_set.ranges.insert(union_set.ranges.end(), it1, ranges.end());
  union_set.ranges.insert(union_set.ranges.end(), it2, other.ranges.end());
  return union_set;
}


template <typename T>
SparseSet<T> SparseSet<T>::operator & (const SparseSet<T> &other) const
{
  SparseSet<T> intersection_set;
  const_iterator it1 = ranges.begin(), it2 = other.ranges.begin();

  while (it1 != ranges.end() && it2 != other.ranges.end()) {
    if (it1->end < it2->begin) {
      // no overlap; *it1 is the smallest
      ++it1;
    } else if (it2->end < it1->begin) {
      // no overlap; *it2 is the smallest
      ++it2;
    } else { // there is overlap, or it1 and it2 are contiguous
      intersection_set.ranges.push_back(range(std::max(it1->begin, it2->begin), std::min(it1->end, it2->end)));

      // continue with earliest end, as multiple ranges may overlap
      // with the latest end.
      if (it1->end < it2->end)
	++it1;
      else
	++it2;
    }
  }

  // ignore the remainder of the set that we have not finished yet

  return intersection_set;
}


template <typename T>
SparseSet<T> &SparseSet<T>::operator += (size_t count)
{
  for (iterator it = ranges.begin(); it != ranges.end(); it++)
    it->begin += count, it->end += count;

  return *this;
}


template <typename T>
SparseSet<T> &SparseSet<T>::operator -= (size_t count)
{
  assert(ranges.size() == 0 || ranges[0].begin >= count);

  for (iterator it = ranges.begin(); it != ranges.end(); it++)
    it->begin -= count, it->end -= count;

  return *this;
}

template <typename T>
SparseSet<T> &SparseSet<T>::operator /= (size_t shrinkFactor)
{
  iterator prev = ranges.end();

  if (shrinkFactor == 1) {
    /* nothing changes */
    return *this;
  }

  for (iterator it = ranges.begin(); it != ranges.end(); it++) {
    it->begin = static_cast<T>(floor(static_cast<double>(it->begin) / shrinkFactor));
    it->end = static_cast<T>(ceil(static_cast<double>(it->end) / shrinkFactor));

    /* The gap between two ranges might have disappeared. The ranges can
       even overlap due to the differences in rounding. */
    if (prev != ranges.end() && prev->end >= it->begin) {
      /* combine tuples */
      it->begin = prev->begin;

      /* it would be invalidated by the erase, so reobtain it */
      it = ranges.erase(prev);
    }

    prev = it;
  }

  return *this;
}

template <typename T>
size_t SparseSet<T>::marshallSize(size_t nrRanges)
{
  return sizeof(uint32_t) + nrRanges * sizeof(range);
}


template <typename T>
ssize_t SparseSet<T>::marshall(void *ptr, size_t maxSize) const
{
  size_t size = marshallSize(ranges.size());

  if (size > maxSize)
    return -1;

  *(uint32_t *) ptr = ranges.size();
  std::memcpy((uint32_t *) ptr + 1, &ranges[0], ranges.size() * sizeof(range));

  return size;
}


template <typename T>
void SparseSet<T>::unmarshall(const void *ptr)
{
  ranges.resize(*(uint32_t *) ptr);
  std::memcpy(&ranges[0], (uint32_t *) ptr + 1, ranges.size() * sizeof(range));
}


template <typename T>
std::ostream &operator << (std::ostream &str, const SparseSet<T> &set)
{
  for (typename SparseSet<T>::const_iterator it = set.getRanges().begin(); it != set.getRanges().end(); it++)
    if (it->end == it->begin + 1)
      str << '[' << it->begin << "] ";
    else
      str << '[' << it->begin << ".." << it->end << "> ";

  return str;
}

#endif


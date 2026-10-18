#include <cstdint>
#include <cstring>
#include <math.h>
#include <omp.h>

namespace pg {
// an element added to atomically, giving what it held
inline int atomic_add(int &x, const int v) {
  int held;
#pragma omp atomic capture
  {
    held = x;
    x += v;
  }
  return held;
}
// the bits of a value as the unsigned integer of its width
inline uint32_t bits(const float x) {
  uint32_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}
inline uint64_t bits(const double x) {
  uint64_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}
// positive infinity in the case's precision
inline fpdtype infinity() { return INFINITY; }
// an element of a view as a body reaches it, and the pointer the arrinfos are read through: the plain ones
template <class T> using Element = T;
using Arrinfos = const ArrInfo *;
} // namespace pg

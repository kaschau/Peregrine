## a unit includes no header: the integer widths are typedefs and the C math is built in
typedef int int32_t;
typedef long long int64_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
namespace pg {
// an element added to atomically, giving what it held
__device__ inline int atomic_add(int &x, const int v) { return atomicAdd(&x, v); }
// the bits of a value as the unsigned integer of its width
__device__ inline uint32_t bits(const float x) { return __float_as_uint(x); }
__device__ inline uint64_t bits(const double x) { return (uint64_t)__double_as_longlong(x); }
// positive infinity in the case's precision
__device__ inline fpdtype infinity() { return (fpdtype)__int_as_float(0x7f800000); }
} // namespace pg

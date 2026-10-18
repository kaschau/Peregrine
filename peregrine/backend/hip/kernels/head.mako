<%include file="peregrine.backend.device.kernels.head"/>
namespace pg {
// a wavefront's lanes, and a value from the lane so many down the wavefront
__device__ inline int lanes() { return warpSize; }
__device__ inline fpdtype shuffle_down(const fpdtype x, const int offset) { return __shfl_down(x, offset); }
// a const element is in the global address space: a flat load through a generic pointer drains every load in flight, a global load overlaps; the arrinfos are read through constant memory
template <class T> struct ElementOf {
  using type = T;
};
template <class T> struct ElementOf<const T> {
  using type = __attribute__((address_space(1))) const T;
};
template <class T> using Element = typename ElementOf<T>::type;
using Arrinfos = __attribute__((address_space(4))) const ArrInfo *;
} // namespace pg

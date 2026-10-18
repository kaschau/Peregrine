<%include file="peregrine.backend.device.kernels.head"/>
namespace pg {
// a warp's lanes, and a value from the lane so many down the warp
__device__ inline int lanes() { return 32; }
__device__ inline fpdtype shuffle_down(const fpdtype x, const int offset) { return __shfl_down_sync(0xffffffff, x, offset); }
// an element of a view as a body reaches it, and the pointer the arrinfos are read through: the plain ones
template <class T> using Element = T;
using Arrinfos = const ArrInfo *;
} // namespace pg

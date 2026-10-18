// every backend's head defines pg::atomic_add, pg::bits, pg::infinity, pg::Element<T> and pg::Arrinfos, and fnqual qualifies every function a kernel calls

namespace pg {
// strides in elements; data is untyped, the rendered kernel knowing the case's precision
struct ArrInfo {
  ${abi.structs["ArrInfo"].cxx("  ")}
};

// the offsets are displacements in cells only the range knows, to the interior cell behind a halo cell or the face on the interface
enum { noffsets = ${noffsets} };
struct Range {
  ${abi.structs["Range"].cxx("  ")}
};

// a tile is of one range's items, so many small entries and a few huge ones both fill the device and the halo edges and corners are never generated
struct Tiling {
  ${abi.structs["Tiling"].cxx("  ")}
};

static_assert(sizeof(ArrInfo) == ${abi.structs["ArrInfo"].size} && sizeof(Range) == ${abi.structs["Range"].size} &&
              sizeof(Tiling) == ${abi.structs["Tiling"].size}, "laid out as the python twins");
}  // namespace pg

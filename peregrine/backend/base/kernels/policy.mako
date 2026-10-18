
namespace pg {

// a tile: its range, the entry the range is of, and its item range
struct Tile {
  int r, e, begin, end;
};
${fnqual} Tile tileof(const Tiling &t, const int rank) {
  const int r = t.range[rank];
  const int begin = (rank - t.first[r]) * t.tile;
  const int n = t.items[r];
  return {r, t.ranges[r].entry, begin, begin + t.tile < n ? begin + t.tile : n};
}

// an item as indices; the layout's fastest index runs fastest
template <int R>
${fnqual} void unravel(int item, const int *extent, int *index) {
% if backend.order == "F":
  for (int d = 0; d < R; d++) {
% else:
  for (int d = R - 1; d >= 0; d--) {
% endif
    index[d] = item % extent[d];
    item /= extent[d];
  }
}
// an item as indices in a fixed axis order, the first running fastest
template <int R, int... O>
${fnqual} void unravel_as(int item, const int *extent, int *index) {
  constexpr int order[] = {O...};
  for (int d = 0; d < R; d++) {
    index[order[d]] = item % extent[order[d]];
    item /= extent[order[d]];
  }
}
// on the F layout a face kernel walks its own axis right after the fastest, so the far-side plane it reads is the one the tile before walked and still in cache
template <int Walk>
${fnqual} void cellat(const Range &c, const int item, int &i,
                                   int &j, int &k) {
  int a[3];
% if backend.order == "F":
  unravel_as<3, 0, Walk, 3 - Walk>(item, c.extent, a);
% else:
  unravel<3>(item, c.extent, a);
% endif
  i = c.start[0] + a[0], j = c.start[1] + a[1], k = c.start[2] + a[2];
}

// the cells of a range, each an Index, walked as cellat says
template <int Walk> struct WalkPolicy {
  ${fnqual} static Index decode(const Range &r, const int item,
                                             const int e) {
    Index at{e, 0, 0, 0, &r};
    cellat<Walk>(r, item, at.i, at.j, at.k);
    return at;
  }
};
// the cells: f()
using CellPolicy = WalkPolicy<1>;
// the faces of one axis, indexed like the cell to the right: f(); the k direction walks k next, the others j
template <int A> using FacePolicy = WalkPolicy<A == 2 ? 2 : 1>;
// one thread per slot of a list array: the Index carries the slot in i until pinning locates the listed cell or finds the slot past the count
struct ListPolicy {
  ${fnqual} static Index decode(const Range &r, const int item,
                                             const int e) {
    return Index{e, item, 0, 0, &r};
  }
};
// an element of the entry's allocation, flat: f(i), the views read flat and never at an index
struct ElementPolicy {
  ${fnqual} static int decode(const Range &, const int item,
                                           const int) {
    return item;
  }
};

} // namespace pg


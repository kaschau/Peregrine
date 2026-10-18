namespace pg {

// a body takes an element by value, const fpdtype x = Q[0]: a head may put a const Element<T> in an address space (hip does), where a const fpdtype & bound to it or a template deduced over it beside a plain fpdtype does not compile
template <class T> ${fnqual} Element<T> &elem(T *p) { return *(Element<T> *)p; }

// where a thread stands: an entry of the table, or an index into one of its ranges; a listed launch's thread keeps its slot beside the cell
struct Entry {
  int e;
};
struct Index {
  ${abi.structs["Index"].cxx("  ")}
};

// extents and strides in elements, read once per tile from the arrinfos through pg::Arrinfos
struct Shape {
  ${abi.structs["Shape"].cxx("  ")}
  ${fnqual} void pin(const ArrInfo *arrinfos, const int e) {
    const Arrinfos r = (Arrinfos)arrinfos + e;
    for (int d = 0; d < 3; d++)
      n[d] = r->extent[d];
    for (int d = 0; d < 5; d++)
      s[d] = r->stride[d];
  }
};

template <int... C> struct Dims {
  static constexpr int count = 1, tail = 1;
};
template <int C0, int... Cs> struct Dims<C0, Cs...> {
  static constexpr int count = C0 * Dims<Cs...>::count, tail = Dims<Cs...>::count;
};

// [l] on the last component dimension is the element, on an outer one the Components inside it
template <class T, int... C> struct Components;
template <class T, int C0> struct Components<T, C0> {
  T *at;
  long stride;
  ${fnqual} Element<T> &operator[](const int l) const {
    return elem(at + l * stride);
  }
};
template <class T, int C0, int C1, int... Cs> struct Components<T, C0, C1, Cs...> {
  T *at;
  long stride;
  ${fnqual} Components<T, C1, Cs...> operator[](const int l) const {
% if backend.order == "F":
    return {at + l * stride, stride * C0};
% else:
    return {at + l * stride, Dims<Cs...>::count};
% endif
  }
};
// the list's layout, read flat: the total count, B segment counts, then segment b's cell codes from start(b), each a cell's code on the allocation's extents
template <class T, int B> struct List {
  T *data;
  const Shape &n;
  const Index &at;
  ${fnqual} Element<T> &operator[](const int i) const { return elem(data + i); }
  ${fnqual} int cells() const { return n.n[0] * n.n[1] * n.n[2]; }
  ${fnqual} int capacity() const { return cells() - B - 1; }
  ${fnqual} int start(const int b) const { return b * cells() + B + 1; }
  // the thread's cell as a code
  ${fnqual} int code() const { return at.i + n.n[0] * (at.j + n.n[1] * at.k); }
  // the thread's slot, once located
  ${fnqual} int slot() const { return at.slot; }
  // the slots run across the segments end to end; false for a slot past the count
  ${fnqual} bool locate(Index &to) const {
    const int slot = to.i;
    to.slot = slot;
    if (slot >= (*this)[0])
      return false;
    int b = 0, first = 0;
    for (; b < B - 1; b++) {
      const int next = first + (*this)[1 + b];
      if (slot < next)
        break;
      first = next;
    }
    int code = (*this)[start(b) + slot - first];
    to.i = code % n.n[0];
    code /= n.n[0];
    to.j = code % n.n[1];
    to.k = code / n.n[1];
    return true;
  }
};

// a view's elements, flat: what a copy or a combination reads
template <class T> struct FlatView {
  T *data;
  ${fnqual} Element<T> &operator[](const int i) const {
    return elem(data + i);
  }
};

// data is pinned once per tile: read through the arrinfo per element it is a pointer chase the compiler cannot hoist past the kernel's stores
template <class T, int... C> struct View {
  ${abi.structs["View"].cxx("  ")}

  ${fnqual} void pin(const Entry &at) {
    const Arrinfos r = (Arrinfos)arrinfos + at.e;
    data = static_cast<T *>(r->data);
  }
  ${fnqual} decltype(auto) atidx(const Shape &n, const long i, const long j,
                                                const long k) const {
    T *p = data + i * n.s[0] + j * n.s[1] + k * n.s[2];
    if constexpr (sizeof...(C) == 0)
      return elem(p);
    else
% if backend.order == "F":
      return Components<T, C...>{p, n.s[3]};
% else:
      return Components<T, C...>{p, Dims<C...>::tail};
% endif
  }
  // at an offset from the index
  ${fnqual} decltype(auto) at(const Shape &n, const Index &at, const int di,
                                           const int dj, const int dk) const {
    return atidx(n, at.i + di, at.j + dj, at.k + dk);
  }
  // an offset fixed in the kernel
  template <int DI, int DJ, int DK>
  ${fnqual} decltype(auto) at(const Shape &n, const Index &at) const {
    return this->at(n, at, DI, DJ, DK);
  }
  // an offset the range carries
  template <int N>
  ${fnqual} decltype(auto) atoffset(const Shape &n, const Index &at) const {
    static_assert(N < noffsets, "a range carries noffsets offsets");
    const int *o = at.range->offset[N];
    return this->at(n, at, o[0], o[1], o[2]);
  }
  ${fnqual} FlatView<T> flat() const { return {data}; }
  // the entry's array as a list over its slots, its components the segments
  ${fnqual} List<T, Dims<C...>::count> list(const Shape &n, const Index &at) const { return {data, n, at}; }
  ${fnqual} decltype(auto) whole(const Shape &n) const {
    if constexpr (sizeof...(C) == 0)
      return elem(data);
    else
% if backend.order == "F":
      return Components<T, C...>{data, n.s[0]};
% else:
      return Components<T, C...>{data, Dims<C...>::tail};
% endif
  }
};

// read through its address, so a device graph sees what the host set since it was built
struct Scalar {
  ${abi.structs["Scalar"].cxx("  ")}
  ${fnqual} fpdtype operator()() const { return *data; }
};

// the twins python builds a kernel's argument from, laid out the same
static_assert(sizeof(View<fpdtype>) == ${abi.structs["View"].size} && sizeof(Shape) == ${abi.structs["Shape"].size} &&
              sizeof(Index) == ${abi.structs["Index"].size} && sizeof(Scalar) == ${abi.structs["Scalar"].size}, "laid out as the python twins");

} // namespace pg

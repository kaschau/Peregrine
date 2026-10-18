<%page args="item"/>
<% threads, waves = backend.bounds %>
## a flat element launch, policy.index None, takes any block and no launch bound
<% bounds = f"__launch_bounds__({threads}, {waves})" if policy.index is not None else "" %>
extern "C" __global__ ${bounds} void pg_${kernel.symbol}(const ${kernel.name} k, const pg::Tiling t) {
  const pg::Tile r = pg::tileof(t, blockIdx.x);
  ${kernel.name} p = k;
  p.pin(pg::Entry{r.e});
  for (int item = r.begin + threadIdx.x; item < r.end; item += blockDim.x) {
${item("", "    ")}\
  }
}

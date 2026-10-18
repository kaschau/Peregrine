<%page args="item"/>
<% threads, waves = backend.bounds %>
## a flat element launch, policy.index None, takes any block and no launch bound
<% bounds = f"__launch_bounds__({threads}, {waves})" if policy.index is not None else "" %>
<% reduce = kernel.reduce %>
<% count = kernel.reduce.count %>
<% identity = f"{reduce.sign}pg::infinity()" %>
extern "C" __global__ ${bounds} void pg_${kernel.symbol}(const ${kernel.name} k, const pg::Tiling t, fpdtype *partials) {
  const pg::Tile r = pg::tileof(t, blockIdx.x);
  ${kernel.name} p = k;
  p.pin(pg::Entry{r.e});
  fpdtype mine[${count}];
  for (int n = 0; n < ${count}; n++)
    mine[n] = ${identity};
  for (int item = r.begin + threadIdx.x; item < r.end; item += blockDim.x) {
${item("mine", "    ")}\
  }
  // the warps fold by shuffle, the block through shared memory sized for 1024 threads at 32 lanes, thread 0 writes
  __shared__ fpdtype warps[32][${count}];
  const int lanes = pg::lanes();
  for (int n = 0; n < ${count}; n++)
    for (int offset = lanes / 2; offset > 0; offset /= 2)
      mine[n] = ${reduce.cfunc}(mine[n], pg::shuffle_down(mine[n], offset));
  if (threadIdx.x % lanes == 0)
    for (int n = 0; n < ${count}; n++)
      warps[threadIdx.x / lanes][n] = mine[n];
  __syncthreads();
  if (threadIdx.x == 0) {
    const int nwarps = (blockDim.x + lanes - 1) / lanes;
    for (int n = 0; n < ${count}; n++) {
      fpdtype v = warps[0][n];
      for (int w = 1; w < nwarps; w++)
        v = ${reduce.cfunc}(v, warps[w][n]);
      partials[blockIdx.x * ${count} + n] = v;
    }
  }
}

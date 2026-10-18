<%page args="item"/>
<% reduce = kernel.reduce %>
<% count = kernel.reduce.count %>
extern "C" __attribute__((visibility("default"))) void pg_${kernel.symbol}(const ${kernel.name} &k, const pg::Tiling &t, fpdtype *out) {
  for (int n = 0; n < ${count}; n++)
    out[n] = ${reduce.sign}pg::infinity();
#pragma omp parallel
  {
    fpdtype mine[${count}];
    for (int n = 0; n < ${count}; n++)
      mine[n] = ${reduce.sign}pg::infinity();
#pragma omp for schedule(static)
    for (int tile = 0; tile < t.tiles; tile++) {
      const pg::Tile r = pg::tileof(t, tile);
      ${kernel.name} p = k;
      p.pin(pg::Entry{r.e});
      for (int item = r.begin; item < r.end; item++) {
${item("mine", "        ")}\
      }
    }
#pragma omp critical
    for (int n = 0; n < ${count}; n++)
      out[n] = ${reduce.cfunc}(out[n], mine[n]);
  }
}

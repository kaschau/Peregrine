<%page args="item"/>
extern "C" __attribute__((visibility("default"))) void pg_${kernel.symbol}(const ${kernel.name} &k, const pg::Tiling &t) {
#pragma omp parallel for schedule(static)
  for (int tile = 0; tile < t.tiles; tile++) {
    const pg::Tile r = pg::tileof(t, tile);
    ${kernel.name} p = k;
    p.pin(pg::Entry{r.e});
    for (int item = r.begin; item < r.end; item++) {
${item("", "      ")}\
    }
  }
}

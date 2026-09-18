# Benchmark methodology

M0 benchmarks infrastructure and deterministic corpus generation. Algorithm
benchmarks will be added as each implementation lands.

## Corpus

`testdata/benchmarks/manifest.json` defines recipes instead of committing large
numeric blobs. Each recipe fixes its generator, seed, row count, dimension, and
shape parameters. The corpus spans:

- low-dimensional Gaussian clusters;
- high-dimensional Gaussian clusters;
- duplicate and equal-distance points;
- noise-heavy mixtures;
- small, medium, and large sizes.

Changing a recipe is a benchmark-contract change. Add a new recipe rather than
silently changing an existing one.

## Running

```sh
go test -run '^$' -bench . -benchmem -count 5 ./... | tee benchmark.txt
```

For statistically meaningful comparisons, use identical hardware and compare
saved outputs with `benchstat`. Record:

- benchmark name and dataset recipe;
- `ns/op`, `B/op`, and `allocs/op`;
- peak process RSS for whole-program benchmarks;
- CPU model and logical CPU count;
- operating system and architecture;
- Go version and effective worker count.

Go's benchmark output covers time and managed allocations, not peak RSS. The CI
benchmark workflow also captures `/usr/bin/time -v` and system metadata. Future
command benchmarks will use this wrapper to measure end-to-end peak RSS.

## CI policy

The benchmark workflow uses the version-pinned `ubuntu-24.04` GitHub runner image
and uploads raw output plus host metadata. Hosted runner hardware can vary, so CI
benchmark numbers are historical signals, not pass/fail thresholds. Release-grade
claims must be reproduced on a named, dedicated host with its CPU governor and
worker count recorded.

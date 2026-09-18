# Benchmark methodology

M0 established the infrastructure and deterministic corpus generation. M1 adds
`BenchmarkMetricDimensions`, which reports metric throughput for dimensions 0,
2, 4, 8, and powers of two through 1024. Run it on each target architecture:

```sh
go test -run '^$' -bench BenchmarkMetricDimensions -benchmem .
```

The bytes/op rate treats both input vectors as read once (16 bytes per
dimension). It is a cross-version throughput indicator, not a claim about
physical memory traffic or cache misses.

`BenchmarkScalarVectorBreakEven` compares the portable scalar kernel with AVX2
or NEON at each dimension. AVX2 is compiled when `GOAMD64=v3`; ARM64 always has
NEON. The measured threshold is applied once by `distance.Select`, outside hot
loops, so vector setup and tail overhead do not penalize short vectors.

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

M4 adds `BenchmarkM4AlgorithmMatrix` for Auto versus forced exact paths and
`BenchmarkApproximateQuality`, which reports ARI, exact-MST edge recall, speedup,
and managed bytes/op. The Auto regression budget is 1.5× the fastest exact
built-in path for each matrix workload; CI records the values rather than using
timing as a flaky pass/fail gate. See [the algorithm guide](algorithms.md).

The pinned M5 release comparison and its explicit runtime/RSS gates are published
in [`benchmarks/m5-linux-amd64.md`](benchmarks/m5-linux-amd64.md).

## CI policy

The benchmark workflow uses the version-pinned `ubuntu-24.04` GitHub runner image
and uploads raw output plus host metadata. Hosted runner hardware can vary, so CI
benchmark numbers are historical signals, not pass/fail thresholds. Release-grade
claims must be reproduced on a named, dedicated host with its CPU governor and
worker count recorded.

# Algorithm benchmark — linux/amd64

Measured on 2026-09-18 with Go 1.24 on an Intel Core i5-12600K using 16 logical
CPUs. Matrix times are medians of three runs at five iterations each with one
worker, which isolates selection overhead from scheduler noise.

| Shape | Algorithm | Time | Allocated bytes/op | Auto / best exact |
| --- | --- | ---: | ---: | ---: |
| 500 × 3 | Auto (k-d tree) | 5.41 ms | 335,304 | 1.31× |
| 500 × 3 | forced k-d tree | 4.12 ms | 335,438 | — |
| 500 × 3 | forced blocked brute force | 5.61 ms | 277,121 | — |
| 500 × 64 | Auto (blocked brute force) | 22.19 ms | 271,856 | 1.00× |
| 500 × 64 | forced blocked brute force | 22.10 ms | 271,856 | — |

Both Auto workloads are within the published 1.5× guardrail. The whole benchmark
process peaked at 91,160 KiB RSS; this includes the Go toolchain and test binary.
Blocked-core scratch is fixed per worker and the streamed MST retains no pairwise
matrix.

The 1,000 × 64 separated-blob quality workload produced:

| ARI | Exact-MST edge recall | Speedup vs exact | Approximate bytes/op |
| ---: | ---: | ---: | ---: |
| 1.000 | 0.1201 | 1.79× | 8,088,432 |

ARI is the primary clustering-quality measure; edge recall is intentionally also
reported because identical flat clusters can arise from different local tree
edges. These figures describe the built-in deterministic four-projection backend,
not a quality guarantee for every dataset.

The 1,000 × 64 blocked workload took 83.9 ms with one worker, 74.4 ms with two,
and 62.4 ms with 16. Parallel speedup is useful but limited because the stable
streamed-MST phase remains serial; goroutine count stays bounded by `Workers`.

Reproduce with:

```sh
/usr/bin/time -v go test -run '^$' \
  -bench 'BenchmarkAlgorithmMatrix|BenchmarkApproximateQuality' \
  -benchmem -benchtime=1x .
```

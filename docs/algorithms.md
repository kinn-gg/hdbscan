# Algorithm selection and approximate mode

`Exact` accepts an `Algorithm` policy in `Config`. The zero value,
`AlgorithmAuto`, deterministically chooses an exact path and never silently
trades quality for speed:

- `AlgorithmKDTree` uses bounded per-worker neighbor heaps for non-empty
  Euclidean vectors with at most 32 dimensions.
- `AlgorithmBruteForce` streams 64×64 distance tiles through the selected
  scalar/SIMD kernel. Scratch is bounded by
  `O(workers*(64²+64*MinSamples))`; no pairwise matrix is retained.
- `AlgorithmApproximate` builds a deterministic four-projection neighbor graph
  and runs a stable Kruskal pass. Its default candidate budget is
  `max(128, 8*MinSamples)`, prioritizing recall over minimum runtime.
- `AlgorithmReference` invokes the quadratic-memory correctness oracle.

Auto considers row count, dimensionality, metric, and deterministic samples of
coordinate spread (a proxy for useful tree pruning). It uses blocked brute force
for custom metrics, high-dimensional or empty vectors, tiny inputs, and samples
where fewer than one quarter of dimensions have nonzero spread. Otherwise it
uses the k-d tree. Forced selection is available for reproducible comparisons.

Every result records the chosen algorithm, worker limit, and approximation flag
in `Result.Metadata`. Approximation must be explicitly requested and
`Metadata.Approximate` is always true for that route. A custom ANN implementation
can replace the built-in graph builder through `Config.ApproximateBackend`.

## Parallelism and memory

Core-distance work uses one bounded worker group; downstream MST and hierarchy
phases do not create nested worker groups. The number of goroutines is therefore
`O(Workers)`, not proportional to point or pair count. Each worker owns its heap
and distance tile, eliminating shared hot counters and false sharing.

## Benchmark contract

The initial Auto guardrail is 1.5× the fastest exact built-in path on the checked-
in low- and high-dimensional matrix. Run:

```sh
go test -run '^$' -bench 'BenchmarkAlgorithmMatrix|BenchmarkBlockedWorkerScaling|BenchmarkApproximateQuality' -benchmem -benchtime=1x
```

`BenchmarkApproximateQuality` reports adjusted Rand index (`ARI`), exact-MST edge
recall, speedup over exact blocked brute force, and managed bytes/op. Wrap the
command in `/usr/bin/time -v` to capture peak RSS. Results are workload- and
machine-specific; checked-in measurements record the environment and should be
updated when selector thresholds or approximate defaults change.

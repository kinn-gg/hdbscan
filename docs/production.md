# Production guide

## API and ownership

`New` validates an immutable `Config`; a `Model` may be shared by goroutines.
`Fit` and `FitPredict` honor context cancellation. Every returned slice belongs
to the caller and never aliases the input or another result. `errors.Is` matches
`ErrInvalidConfig`, `ErrTooFewPoints`, and context cancellation errors.

`Result.Extract` reuses the condensed hierarchy and does not repeat distance,
neighbor, or MST work. `WriteMST`, `WriteSingleLinkageTree`, and
`WriteCondensedTree` stream CSV or newline-delimited JSON to an `io.Writer`.

## Complexity and memory sizing

The reference path costs O(n²) time and memory. Exact k-d-tree and blocked paths
avoid an n-by-n matrix: retained outputs are O(n), input is O(n*d), and working
memory is O(n + workers*min_samples) for k-d-tree queries or bounded blocks for
brute force. Worst-case exact runtime remains O(n²). Results retain three tree
edge sets and four point/cluster vectors; budget roughly a few hundred bytes per
observation in addition to the input and worker scratch.

## Reproducibility

Results are deterministic for a fixed Go version, architecture, input order,
configuration, and metric. Equal edges use stable point-index tie breaks.
Vector kernels can reassociate floating-point sums across architectures; see
[numerics](numerics.md). Approximate results are always marked in metadata and
must not be compared as though they were exact.

## Profiling

Run CPU and allocation profiles against a representative benchmark:

```sh
go test -run '^$' -bench BenchmarkExactLowDimensional -cpuprofile cpu.pprof -memprofile mem.pprof
go tool pprof -http=:0 cpu.pprof
go tool pprof -http=:0 mem.pprof
```

Use `/usr/bin/time -v` around a benchmark binary to capture peak RSS. Pin CPU
governor, Go version, worker count, and dataset recipe before comparing runs.

## Support and licensing

v0.1 covers dense finite float64 observations, exact Euclidean/Manhattan/
Minkowski metrics, dense precomputed distances through `ReferencePrecomputed`,
EOM/leaf extraction, probabilities, persistence, GLOSH scores, and tree export.
Prediction of unseen samples and soft membership are planned after v0.1.

The module is BSD-3-Clause. `LICENSE` and `NOTICE` retain attribution to the
upstream `scikit-learn-contrib/hdbscan` project used as the compatibility oracle.

# Production guide

## API and ownership

`New` validates an immutable `Config`; a `Model` may be shared by goroutines.
`Fit` and `FitPredict` honor context cancellation. Every returned slice belongs
to the caller and never aliases the input or another result. `errors.Is` matches
`ErrInvalidConfig`, `ErrTooFewPoints`, and context cancellation errors.

`Result.Extract` reuses the condensed hierarchy and does not repeat distance,
neighbor, or MST work. `WriteMST`, `WriteSingleLinkageTree`, and
`WriteCondensedTree` stream CSV or newline-delimited JSON to an `io.Writer`.

Prediction state is retained only when `Config.PredictionData` is true. It owns
a copy of the finite dense training data, self-inclusive prediction core
distances, selected-cluster maps, and exemplar indices. `ApproximatePredict`
returns stable fitted labels and strengths; `ApproximatePredictScores` estimates
new-point GLOSH scores. `MembershipVector`, `MembershipVectors`, and
`AllPointsMembershipVectors` provide soft membership. Their `Into` variants let
callers reuse output buffers; a batch uses scratch proportional to training rows
and cluster count, not query count, and performs no per-query heap allocation.

Novel points that do not enter a selected cluster are labeled `-1` with zero
strength. A model with no selected clusters predicts every novel point as noise;
an allowed single-cluster model returns its sole membership column. Non-finite or
wrong-width query matrices are rejected. Prediction is unavailable for models
fit from precomputed distances because they have no feature vectors.

FLASC state is independently retained with `Config.BranchDetectionData`.
`BranchCore` retains O(n*min_samples) neighbor indices; `BranchFull` can emit
O(n²) edges for a dense cluster. Packed graphs store one uint64 and two float64
weights per edge and stream through `BranchGraph.Write`. See
[branch detection](branch-detection.md) before enabling it on untrusted or
unbounded input sizes.

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

The package covers dense finite float64 observations, exact Euclidean/Manhattan/
Minkowski metrics, dense precomputed distances through `ReferencePrecomputed`,
EOM/leaf extraction, probabilities, persistence, GLOSH scores, and tree export.
Dense fits can additionally retain opt-in prediction and soft-membership state.

The module is BSD-3-Clause. `LICENSE` and `NOTICE` retain attribution to the
upstream `scikit-learn-contrib/hdbscan` project used as the compatibility oracle.
The stable compatibility and architecture policy is in [the v1 contract](v1.md).

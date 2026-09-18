# hdbscan

A deterministic, cancellable, memory-conscious Go implementation of HDBSCAN\*.

Development is organized by [implementation milestones](MILESTONES.md).
`Exact` selects an exact, memory-efficient implementation for the data shape and
metric. `Reference` remains the quadratic-memory correctness oracle.

```go
model, err := hdbscan.New(hdbscan.Config{
    MinClusterSize: 5,
    MinSamples: 5,
    ClusterSelectionMethod: hdbscan.EOM,
})
if err != nil { /* handle invalid configuration */ }
result, err := model.Fit(ctx, data)
```

Set `PredictionData: true` before fitting to retain the data needed for
out-of-sample assignment and soft clustering. Prediction is opt-in so ordinary
fits do not retain a copy of the input:

```go
result, err := hdbscan.Fit(ctx, training, hdbscan.Config{
    MinClusterSize: 5,
    PredictionData: true,
})
labels, strengths, err := result.ApproximatePredict(ctx, novel)
memberships, err := result.MembershipVectors(ctx, novel)
```

Caller-buffer forms (`PredictInto`, `PredictScoresInto`,
`MembershipVectorsInto`, and `AllPointsMembershipVectorsInto`) keep output and
scratch memory bounded for large batches.

For repeatable benchmarks, set `Config.Algorithm` to `AlgorithmKDTree` or
`AlgorithmBruteForce`. `AlgorithmApproximate` is opt-in and is always identified
by `result.Metadata.Approximate`; `Auto` never selects it.

`Exact` never materializes the pairwise distance matrix. See the
[algorithm guide](docs/algorithms.md). `Reference` is limited to smaller data sets;
see the [reference implementation guide](docs/reference.md).

Every result owns its slices. `Result.Extract` can apply alternate EOM/leaf,
single-cluster, size, epsilon, and persistence settings to the retained hierarchy
without rebuilding neighbors or the MST. Tree exporters stream CSV or newline-
delimited JSON directly to an `io.Writer`.

Extended parity includes Chebyshev, Canberra, and Bray-Curtis metrics,
caller-owned CSR distance graphs through `ReferenceSparse`, robust single
linkage with reusable hierarchy cuts, and the streaming `ValidityIndex` DBCV
implementation. Sparse inputs are never densified; disconnected graphs return
`ErrDisconnected` with an explicit component count.

FLASC branch detection is opt-in through `Config.BranchDetectionData` and
`Result.DetectBranches`; it includes packed approximation graphs, per-cluster
branch hierarchies and persistence, membership strengths, streaming graph export,
and approximate branch prediction. See the [branch guide](docs/branch-detection.md)
and [v1 compatibility policy](docs/v1.md).

See the [production guide](docs/production.md) for complexity, memory sizing,
profiling, cancellation, reproducibility, and release support.

## Development

```sh
go test ./...
go test -race ./...
go test -run '^$' -bench . -benchmem ./...
```

Python parity fixtures are reproducible with `uv`:

```sh
uv sync --locked --project tools/fixtures
uv run --project tools/fixtures python tools/fixtures/generate.py --check
```

See [the compatibility contract](docs/compatibility.md) and
[benchmarking guide](docs/benchmarking.md).

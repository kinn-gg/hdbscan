# hdbscan

A production-oriented Go implementation of HDBSCAN\*: density-based clustering
that discovers clusters of varying shape and labels low-density observations as
noise without requiring the number of clusters in advance.

The project aims to provide:

- deterministic results for a fixed input and configuration;
- exact clustering with bounded, subquadratic auxiliary memory;
- explicit cancellation and ownership semantics for production Go programs;
- compatibility with the core algorithms and outputs of Python's `hdbscan`.

## Features

- Dense vectors, precomputed distance matrices, and sparse CSR distance graphs
- Exact k-d tree and blocked brute-force execution with automatic selection
- EOM and leaf cluster selection, probabilities, persistence, and outlier scores
- Out-of-sample prediction and soft membership vectors
- Euclidean, Manhattan, Minkowski, Chebyshev, Canberra, Bray-Curtis, and custom
  metrics
- Robust single linkage, DBCV validity scoring, and FLASC branch detection
- Streaming CSV and JSON export for trees and graphs
- Context cancellation, deterministic tie-breaking, and pure Go fallbacks

Approximate execution and memory-heavy prediction or branch data are always
opt-in. The default exact path does not materialize a pairwise distance matrix.

## Quick start

```sh
go get github.com/kinn-gg/hdbscan
```

```go
model, err := hdbscan.New(hdbscan.Config{
    MinClusterSize: 5,
    MinSamples:     5,
})
if err != nil {
    return err
}

result, err := model.Fit(ctx, data)
if err != nil {
    return err
}

fmt.Println(result.Labels)
```

Enable `PredictionData` when fitting to support assignment and soft membership
for new observations:

```go
result, err := hdbscan.Fit(ctx, training, hdbscan.Config{
    MinClusterSize: 5,
    PredictionData: true,
})
labels, strengths, err := result.ApproximatePredict(ctx, novel)
```

## Documentation

- [Algorithms and execution modes](docs/algorithms.md)
- [Compatibility](docs/compatibility.md)
- [Prediction and production guidance](docs/production.md)
- [Branch detection](docs/branch-detection.md)
- [Version 1 support policy](docs/v1.md)
- [Benchmark methodology](docs/benchmarking.md)

## Development

```sh
go test ./...
go test -race ./...
golangci-lint run
go test -run '^$' -bench . -benchmem ./...
```

Python parity fixtures are reproducible with `uv`:

```sh
uv sync --locked --project tools/fixtures
uv run --project tools/fixtures python tools/fixtures/generate.py --check
```

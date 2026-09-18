# hdbscan

An in-progress, memory-efficient Go implementation of HDBSCAN\*.

Development is organized by [implementation milestones](MILESTONES.md). The
default low- and moderate-dimensional Euclidean route is the exact,
memory-efficient `Exact` engine. `Reference` remains the quadratic correctness
oracle and fallback for unsupported tree workloads.

```go
result, err := hdbscan.Exact(ctx, data, hdbscan.Config{
    MinClusterSize: 5,
    MinSamples: 5,
    ClusterSelectionMethod: hdbscan.EOM,
})
```

`Exact` never materializes the pairwise distance matrix. See the
[exact-engine guide](docs/exact.md). `Reference` is limited to smaller data sets;
see the [reference implementation guide](docs/reference.md).

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

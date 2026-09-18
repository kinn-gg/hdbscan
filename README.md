# hdbscan

An in-progress, memory-efficient Go implementation of HDBSCAN\*.

Development is organized by [implementation milestones](MILESTONES.md). The
current milestone provides an exact, deterministic `Reference` HDBSCAN* pipeline
for correctness and parity testing, built on the validated distance kernels.

```go
result, err := hdbscan.Reference(ctx, data, hdbscan.Config{
    MinClusterSize: 5,
    MinSamples: 5,
    ClusterSelectionMethod: hdbscan.EOM,
})
```

`Reference` intentionally materializes the pairwise distance matrix and is
limited to smaller data sets. See [the reference implementation guide](docs/reference.md).

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

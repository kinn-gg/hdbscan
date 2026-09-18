# hdbscan

An in-progress, memory-efficient Go implementation of HDBSCAN\*.

Development is organized by [implementation milestones](MILESTONES.md). The
current milestone establishes a reproducible compatibility and performance
baseline against `scikit-learn-contrib/hdbscan` before the algorithm is ported.

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

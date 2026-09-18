# Release baseline benchmark — Linux amd64

Recorded 2026-09-18 on an Intel Core i5-12600K (10 cores, 16 logical CPUs),
Linux 7.0.0-31-generic, Go 1.27.1, and upstream `hdbscan` 0.8.44. The primary
workload has 2,000 uniformly distributed three-dimensional points,
`min_cluster_size=10`, `min_samples=10`, exact execution, and the host's 16
logical CPUs available. Each implementation was warmed once before timing.

| Implementation | Runtime | Peak RSS | Gate relative to Python |
|---|---:|---:|---:|
| Python generic exact | 95.3 ms | 193,336 KiB | baseline |
| Go Auto exact | 44.6 ms | 92,028 KiB | 0.47x time, 0.48x RSS |
| Release target | <=119.1 ms | <=145,002 KiB | <=1.25x time, <=0.75x RSS |

The run meets both initial release gates. Peak RSS is whole-process `/usr/bin/time -v`
and therefore conservatively includes each language runtime and, for Go, the test
harness. Go's benchmark reported 1,418,992 B/op and 24,551 allocations/op. Hosted
CI continues to publish measurements without silently weakening the release gate;
release claims must be rerun on this named host class or another pinned host and
committed as a new report.

Commands:

```sh
go test -run '^$' -bench BenchmarkExactLowDimensional -benchmem -benchtime=1x .
/usr/bin/time -v go test -run '^$' -bench BenchmarkExactLowDimensional -benchtime=1x -count=1 .
/usr/bin/time -v tools/fixtures/.venv/bin/python -c '... HDBSCAN(...).fit(x)'
```

The Go and NumPy generators use the same seed and distribution but different RNG
implementations; this is a shape-level release gate rather than a per-point parity
comparison. Correctness remains covered by the versioned shared parity corpus.

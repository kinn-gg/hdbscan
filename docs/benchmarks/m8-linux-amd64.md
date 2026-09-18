# M8 FLASC benchmark — linux/amd64

Recorded 2026-09-18 on an Intel Core i5-12600K (16 logical CPUs) using the
repository's deterministic 36-row, two-dimensional three-flare workload.

```text
BenchmarkBranchDetection-16  18823 ns/op  17720 B/op  222 allocs/op
BenchmarkBranchDetection-16  15758 ns/op  17720 B/op  222 allocs/op
BenchmarkBranchDetection-16  14333 ns/op  17720 B/op  222 allocs/op
```

Command: `go test -run '^$' -bench '^BenchmarkBranchDetection$' -benchmem
-count=3`. The fit and opt-in branch data are prepared outside the timed region;
the benchmark measures full-graph branch detection and owned result allocation.

The package and tests are cross-compiled for linux/arm64, including the portable
and NEON-dispatch source. No physical arm64 host was available for this record,
so no arm64 performance result is claimed. No float32, GPU, mmap, external-memory,
or parallel-linkage accelerator participates in correctness or this benchmark.

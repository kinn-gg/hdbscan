# Exact-engine benchmark — linux/amd64

Measured on 2026-09-18 with Go 1.24 on an Intel Core i5-12600K. Each row is one
iteration of the checked-in benchmark; peak RSS includes the Go test process and
toolchain overhead.

| Engine | Shape | Time | Allocated bytes/op | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| `Exact` | 2,000 x 3 | 37.2 ms | 1,420,872 | — |
| `Reference` | 2,000 x 3 | 223 ms | 32,996,912 | 91,792 KiB |
| `Exact` | 20,000 x 3 | 3.37 s | 14,780,424 | 91,360 KiB |

At equal shape, `Exact` is about 6x faster and reports 23x fewer allocated
bytes. The 10x-larger exact workload remains just below the measured peak RSS of
the reference workload, demonstrating that point growth does not hide an
`O(n²)` allocation. Runtime at 20,000 points reflects the stable streamed-Prim
canonicalization, whose memory is linear but whose time is quadratic.

Reproduce allocation and wall-time measurements with:

```sh
go test -run '^$' -bench 'Benchmark(Exact|Reference)LowDimensional' -benchmem -benchtime=1x
go test -run '^$' -bench BenchmarkExactTenTimesLarger -benchmem -benchtime=1x
```

Wrap either command in `/usr/bin/time -v` to record maximum resident set size.

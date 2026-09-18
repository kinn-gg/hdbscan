# M1 metric throughput: Linux amd64

Recorded 2026-09-18 on Linux amd64 with Go 1.27.1 and an Intel Core i5-12600K.
The selected implementation was `scalar`. Results are a single local sample;
repeat with `-count 5` and benchstat before using them for regressions.

| Dimensions | ns/op | MB/s | B/op | allocs/op |
|---:|---:|---:|---:|---:|
| 0 | 1.916 | n/a | 0 | 0 |
| 2 | 3.674 | 8,709 | 0 | 0 |
| 4 | 7.103 | 9,010 | 0 | 0 |
| 8 | 12.02 | 10,646 | 0 | 0 |
| 16 | 22.68 | 11,289 | 0 | 0 |
| 32 | 36.86 | 13,889 | 0 | 0 |
| 64 | 65.49 | 15,637 | 0 | 0 |
| 128 | 123.9 | 16,524 | 0 | 0 |
| 256 | 241.8 | 16,937 | 0 | 0 |
| 512 | 471.8 | 17,363 | 0 | 0 |
| 1024 | 970.3 | 16,886 | 0 | 0 |

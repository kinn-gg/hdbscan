# M1 metric throughput: Darwin ARM64

Recorded 2026-09-18 by GitHub Actions on its macOS 15 ARM64 runner with Go
1.25.x and an Apple M1 (Virtual) CPU. The 100 ms samples compare squared
Euclidean kernels directly. Both implementations reported 0 B/op and 0
allocations/op.

The results establish the ARM64 dispatch threshold at 64 dimensions. NEON is
slower through 32 dimensions, then faster from 64 onward.

| Dimensions | Scalar ns/op | NEON ns/op | Speedup |
|---:|---:|---:|---:|
| 2 | 4.982 | 9.727 | 0.51x |
| 4 | 6.193 | 10.36 | 0.60x |
| 8 | 7.905 | 13.22 | 0.60x |
| 16 | 13.48 | 21.72 | 0.62x |
| 32 | 30.95 | 32.93 | 0.94x |
| 64 | 66.62 | 51.46 | 1.29x |
| 128 | 160.6 | 105.9 | 1.52x |
| 256 | 421.0 | 206.4 | 2.04x |
| 512 | 846.4 | 385.5 | 2.20x |
| 1024 | 1933 | 746.9 | 2.59x |

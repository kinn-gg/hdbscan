//go:build arm64 || (amd64 && amd64.v3)

package distance

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func TestVectorSelectionThreshold(t *testing.T) {
	k := vectorKernelsForTest()
	if got := Select(4096).Name; got != k.Name {
		t.Fatalf("Select(4096) = %q, want %q", got, k.Name)
	}
	if got := Select(0).Name; got != "scalar" {
		t.Fatalf("Select(0) = %q, want scalar", got)
	}
}

func TestVectorKernelParity(t *testing.T) {
	for _, dims := range []int{0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 511} {
		a, b := make([]float64, dims), make([]float64, dims)
		for i := range a {
			a[i] = rand.Float64()*2e3 - 1e3
			b[i] = rand.Float64()*2e3 - 1e3
		}
		k := vectorKernelsForTest()
		assertClose(t, "dot", dims, k.Dot(a, b), Dot(a, b))
		assertClose(t, "squared Euclidean", dims, k.SquaredEuclidean(a, b), SquaredEuclidean(a, b))
	}
}

func TestVectorBlockParityAndAllocations(t *testing.T) {
	const rows, dims = 67, 35
	a, b := make([]float64, rows*dims), make([]float64, rows*dims)
	for i := range a {
		a[i], b[i] = float64(i%19)-9, float64(i%23)-11
	}
	want, got := make([]float64, rows*rows), make([]float64, rows*rows)
	SquaredEuclideanBlock(want, a, b, rows, rows, dims)
	k := vectorKernelsForTest()
	k.SquaredEuclideanBlock(got, a, b, rows, rows, dims)
	for i := range want {
		assertClose(t, "block", i, got[i], want[i])
	}
	if allocs := testing.AllocsPerRun(100, func() { k.SquaredEuclideanBlock(got, a, b, rows, rows, dims) }); allocs != 0 {
		t.Fatalf("vector block allocated %g times", allocs)
	}
}

func FuzzVectorKernelParity(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4}, []byte{4, 3, 2, 1})
	f.Fuzz(func(t *testing.T, x, y []byte) {
		if len(x) != len(y) || len(x) > 4096 {
			t.Skip()
		}
		a, b := make([]float64, len(x)), make([]float64, len(x))
		for i := range x {
			a[i], b[i] = float64(int8(x[i])), float64(int8(y[i]))
		}
		k := vectorKernelsForTest()
		assertClose(t, "dot", len(a), k.Dot(a, b), Dot(a, b))
		assertClose(t, "squared Euclidean", len(a), k.SquaredEuclidean(a, b), SquaredEuclidean(a, b))
	})
}

func assertClose(t *testing.T, name string, n int, got, want float64) {
	t.Helper()
	tolerance := 2e-14 * math.Max(1, math.Abs(want))
	if math.Abs(got-want) > tolerance {
		t.Fatalf("%s[%d] = %.17g, scalar %.17g (tolerance %.3g)", name, n, got, want, tolerance)
	}
}

func BenchmarkScalarVectorBreakEven(b *testing.B) {
	vector := vectorKernelsForTest()
	for _, dims := range []int{2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} {
		a, c := make([]float64, dims), make([]float64, dims)
		for i := range a {
			a[i], c[i] = float64(i%17), float64(i%13)
		}
		for _, kernel := range []Kernels{scalarKernels, vector} {
			b.Run(fmt.Sprintf("%s/dims=%d", kernel.Name, dims), func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(16 * dims))
				for i := 0; i < b.N; i++ {
					_ = kernel.SquaredEuclidean(a, c)
				}
			})
		}
	}
}

package distance

import (
	"math"
	"testing"
)

func TestScalarKernels(t *testing.T) {
	a, b := []float64{1, -2, 3}, []float64{4, 2, -1}
	if got := SquaredEuclidean(a, b); got != 41 {
		t.Errorf("SquaredEuclidean = %g", got)
	}
	if got := Euclidean(a, b); math.Abs(got-math.Sqrt(41)) > 1e-15 {
		t.Errorf("Euclidean = %g", got)
	}
	if got := Manhattan(a, b); got != 11 {
		t.Errorf("Manhattan = %g", got)
	}
	if got := Dot(a, b); got != -3 {
		t.Errorf("Dot = %g", got)
	}
}

func TestSquaredEuclideanBlock(t *testing.T) {
	a := []float64{0, 0, 1, 1}
	b := []float64{1, 0, 2, 2}
	dst := make([]float64, 4)
	SquaredEuclideanBlock(dst, a, b, 2, 2, 2)
	want := []float64{1, 8, 1, 2}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst = %v, want %v", dst, want)
		}
	}
	if allocs := testing.AllocsPerRun(1000, func() { SquaredEuclideanBlock(dst, a, b, 2, 2, 2) }); allocs != 0 {
		t.Fatalf("block kernel allocated %g times", allocs)
	}
}

func TestSelectedKernelsMatchScalar(t *testing.T) {
	k := Select(3)
	a, b := []float64{1, 2, 3}, []float64{-2, 4, 8}
	if got, want := k.Dot(a, b), Dot(a, b); got != want {
		t.Fatalf("selected dot = %g, scalar = %g", got, want)
	}
	if got, want := k.SquaredEuclidean(a, b), SquaredEuclidean(a, b); got != want {
		t.Fatalf("selected distance = %g, scalar = %g", got, want)
	}
}

func FuzzScalarDistanceSymmetry(f *testing.F) {
	f.Add([]byte{1, 2}, []byte{3, 4})
	f.Fuzz(func(t *testing.T, x, y []byte) {
		if len(x) != len(y) || len(x) > 2048 {
			t.Skip()
		}
		a, b := make([]float64, len(x)), make([]float64, len(x))
		for i := range x {
			a[i], b[i] = float64(int8(x[i])), float64(int8(y[i]))
		}
		if ab, ba := SquaredEuclidean(a, b), SquaredEuclidean(b, a); ab != ba {
			t.Fatalf("asymmetric: %g != %g", ab, ba)
		}
	})
}

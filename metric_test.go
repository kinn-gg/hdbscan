package hdbscan

import (
	"fmt"
	"math"
	"testing"
)

func TestMetricsReferenceValues(t *testing.T) {
	a, b := []float64{1, -2, 3}, []float64{4, 2, -1}
	tests := []struct {
		metric Metric
		want   float64
	}{
		{SquaredEuclidean, 41},
		{Euclidean, math.Sqrt(41)},
		{Manhattan, 11},
	}
	m3, err := NewMinkowski(3)
	if err != nil {
		t.Fatal(err)
	}
	tests = append(tests, struct {
		metric Metric
		want   float64
	}{m3, math.Cbrt(155)})
	for _, test := range tests {
		if got := test.metric.Distance(a, b); math.Abs(got-test.want) > 2e-15*test.want {
			t.Errorf("%s = %.17g, want %.17g", test.metric, got, test.want)
		}
		if got := test.metric.Distance(nil, nil); got != 0 {
			t.Errorf("%s empty = %g", test.metric, got)
		}
	}
}

func TestMetricExtremeValues(t *testing.T) {
	if got := SquaredEuclidean.Distance([]float64{1e-150}, []float64{0}); got != 1e-300 {
		t.Fatalf("small = %.17g", got)
	}
	if got := Euclidean.Distance([]float64{1e150}, []float64{0}); got != 1e150 {
		t.Fatalf("large = %.17g", got)
	}
	if got := Euclidean.Distance([]float64{1e200}, []float64{0}); got != 1e200 {
		t.Fatalf("overflow-resistant = %.17g", got)
	}
	if got := Euclidean.Distance([]float64{1e-200}, []float64{0}); got != 1e-200 {
		t.Fatalf("underflow-resistant = %.17g", got)
	}
	if got := Manhattan.Distance([]float64{7, 7}, []float64{7, 7}); got != 0 {
		t.Fatalf("duplicate = %g", got)
	}
}

func TestBuiltInMetricsAllocateZero(t *testing.T) {
	a, b := make([]float64, 64), make([]float64, 64)
	m3, err := NewMinkowski(3)
	if err != nil {
		t.Fatal(err)
	}
	metrics := []Metric{SquaredEuclidean, Euclidean, Manhattan, m3}
	for _, metric := range metrics {
		if allocs := testing.AllocsPerRun(1000, func() { _ = metric.Distance(a, b) }); allocs != 0 {
			t.Errorf("%s allocated %g times", metric, allocs)
		}
	}
}

func FuzzEuclideanAgreement(f *testing.F) {
	f.Add([]byte{1, 2, 3}, []byte{3, 2, 1})
	f.Fuzz(func(t *testing.T, x, y []byte) {
		if len(x) != len(y) || len(x) > 1024 {
			t.Skip()
		}
		a, b := make([]float64, len(x)), make([]float64, len(x))
		for i := range x {
			a[i] = float64(int8(x[i]))
			b[i] = float64(int8(y[i]))
		}
		d := Euclidean.Distance(a, b)
		sq := SquaredEuclidean.Distance(a, b)
		if math.Abs(d*d-sq) > 1e-12*math.Max(1, sq) {
			t.Fatalf("euclidean disagreement: %g^2 != %g", d, sq)
		}
	})
}

func BenchmarkMetricDimensions(b *testing.B) {
	for _, dims := range []int{0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} {
		a, c := make([]float64, dims), make([]float64, dims)
		for i := range a {
			a[i], c[i] = float64(i), float64(i+1)
		}
		b.Run(fmt.Sprintf("Euclidean/dims=%d", dims), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(16 * dims))
			for i := 0; i < b.N; i++ {
				_ = Euclidean.Distance(a, c)
			}
		})
	}
}

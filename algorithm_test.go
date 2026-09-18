package hdbscan

import (
	"context"
	"fmt"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/kinn-gg/hdbscan/internal/evaluation"
)

func TestSelectAlgorithm(t *testing.T) {
	if got := SelectAlgorithm(randomDense(100, 3, 1), Euclidean); got != AlgorithmKDTree {
		t.Fatalf("low-dimensional selection = %q", got)
	}
	if got := SelectAlgorithm(randomDense(100, 64, 1), Euclidean); got != AlgorithmBruteForce {
		t.Fatalf("high-dimensional selection = %q", got)
	}
	degenerate := Dense64{Data: make([]float64, 100*8), Rows: 100, Cols: 8}
	if got := SelectAlgorithm(degenerate, Euclidean); got != AlgorithmBruteForce {
		t.Fatalf("degenerate selection = %q", got)
	}
}

func TestForcedExactAlgorithmsMatchReference(t *testing.T) {
	x := randomDense(75, 6, 44)
	want, err := Reference(context.Background(), x, Config{MinClusterSize: 3, MinSamples: 5})
	if err != nil {
		t.Fatal(err)
	}
	for _, algorithm := range []Algorithm{AlgorithmKDTree, AlgorithmBruteForce} {
		got, err := Exact(context.Background(), x, Config{MinClusterSize: 3, MinSamples: 5, Algorithm: algorithm, Workers: 2})
		if err != nil {
			t.Fatalf("%s: %v", algorithm, err)
		}
		if got.Metadata.Algorithm != algorithm || got.Metadata.Approximate {
			t.Fatalf("metadata = %+v", got.Metadata)
		}
		got.Metadata, want.Metadata = Metadata{}, Metadata{}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s differs from reference", algorithm)
		}
	}
}

func TestBruteForceCustomMetricMatchesReference(t *testing.T) {
	x := randomDense(45, 7, 12)
	cfg := Config{MinClusterSize: 3, MinSamples: 4, Metric: Manhattan}
	want, err := Reference(context.Background(), x, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Algorithm = AlgorithmBruteForce
	got, err := Exact(context.Background(), x, cfg)
	if err != nil {
		t.Fatal(err)
	}
	got.Metadata, want.Metadata = Metadata{}, Metadata{}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("custom-metric brute force differs from reference")
	}
}

func TestApproximateIsExplicitAndLabeled(t *testing.T) {
	x := randomDense(80, 12, 9)
	auto, err := Exact(context.Background(), x, Config{MinClusterSize: 3})
	if err != nil {
		t.Fatal(err)
	}
	if auto.Metadata.Approximate {
		t.Fatal("Auto selected approximate execution")
	}
	got, err := Exact(context.Background(), x, Config{MinClusterSize: 3, Algorithm: AlgorithmApproximate, Workers: 2})
	if err != nil {
		t.Fatal(err)
	}
	if !got.Metadata.Approximate || got.Metadata.Algorithm != AlgorithmApproximate {
		t.Fatalf("metadata = %+v", got.Metadata)
	}
	if len(got.MinimumSpanningTree) != x.Rows-1 {
		t.Fatalf("MST edges = %d", len(got.MinimumSpanningTree))
	}
}

type recordingBackend struct{ called bool }

func (b *recordingBackend) Build(ctx context.Context, x Dense64, metric Metric, k int, alpha float64, workers int) ([]float64, []MSTEdge, error) {
	b.called = true
	core, err := blockedCoreDistances(ctx, x, metric, k, workers)
	if err != nil {
		return nil, nil, err
	}
	mst, err := streamedPrim(ctx, x, metric, core, alpha)
	return core, mst, err
}

func TestReplaceableApproximateBackend(t *testing.T) {
	b := &recordingBackend{}
	_, err := Exact(context.Background(), randomDense(20, 3, 4), Config{MinClusterSize: 2, Algorithm: AlgorithmApproximate, ApproximateBackend: b})
	if err != nil {
		t.Fatal(err)
	}
	if !b.called {
		t.Fatal("backend was not called")
	}
}

func TestInvalidAlgorithm(t *testing.T) {
	_, err := Exact(context.Background(), randomDense(10, 2, 1), Config{MinClusterSize: 2, Algorithm: "bogus"})
	if err == nil {
		t.Fatal("expected error")
	}
}

type invalidBackend struct{}

func (invalidBackend) Build(context.Context, Dense64, Metric, int, float64, int) ([]float64, []MSTEdge, error) {
	return nil, nil, nil
}

func TestInvalidApproximateBackendResult(t *testing.T) {
	_, err := Exact(context.Background(), randomDense(10, 2, 1), Config{MinClusterSize: 2, Algorithm: AlgorithmApproximate, ApproximateBackend: invalidBackend{}})
	if err == nil {
		t.Fatal("expected backend validation error")
	}
}

func mstEdgeRecall(got, want []MSTEdge) float64 {
	wanted := make(map[[2]int]bool, len(want))
	for _, e := range want {
		if e.From > e.To {
			e.From, e.To = e.To, e.From
		}
		wanted[[2]int{e.From, e.To}] = true
	}
	var hits int
	for _, e := range got {
		if e.From > e.To {
			e.From, e.To = e.To, e.From
		}
		if wanted[[2]int{e.From, e.To}] {
			hits++
		}
	}
	return float64(hits) / float64(len(want))
}

func BenchmarkAlgorithmMatrix(b *testing.B) {
	for _, shape := range []struct {
		name string
		n, d int
	}{{"low_dim", 500, 3}, {"high_dim", 500, 64}} {
		x := randomDense(shape.n, shape.d, 2026)
		for _, algorithm := range []Algorithm{AlgorithmAuto, AlgorithmKDTree, AlgorithmBruteForce, AlgorithmApproximate} {
			if algorithm == AlgorithmKDTree && shape.d > kdTreeMaxDimensions {
				continue
			}
			b.Run(shape.name+"/"+string(algorithm), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if _, err := Exact(context.Background(), x, Config{MinClusterSize: 10, MinSamples: 10, Workers: 1, Algorithm: algorithm}); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func BenchmarkBlockedWorkerScaling(b *testing.B) {
	x := randomDense(1000, 64, 2027)
	for _, workers := range []int{1, 2, runtime.GOMAXPROCS(0)} {
		b.Run(fmt.Sprintf("workers_%d", workers), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := Exact(context.Background(), x, Config{MinClusterSize: 10, MinSamples: 10, Workers: workers, Algorithm: AlgorithmBruteForce}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkApproximateQuality(b *testing.B) {
	// Separated high-dimensional blobs make the clustering-quality metric
	// meaningful while MST recall still measures neighborhood fidelity.
	x := randomDense(1000, 64, 2026)
	for row := 0; row < x.Rows; row++ {
		cluster := row % 5
		for col := 0; col < x.Cols; col++ {
			x.Data[row*x.Cols+col] = x.Data[row*x.Cols+col]*0.025 + float64(cluster*8)
		}
	}
	cfg := Config{MinClusterSize: 10, MinSamples: 10, Workers: runtime.GOMAXPROCS(0), Algorithm: AlgorithmBruteForce}
	start := time.Now()
	exact, err := Exact(context.Background(), x, cfg)
	if err != nil {
		b.Fatal(err)
	}
	exactTime := time.Since(start)
	cfg.Algorithm = AlgorithmApproximate
	start = time.Now()
	approximate, err := Exact(context.Background(), x, cfg)
	if err != nil {
		b.Fatal(err)
	}
	approximateTime := time.Since(start)
	ari := evaluation.AdjustedRandIndex(exact.Labels, approximate.Labels)
	recall := mstEdgeRecall(approximate.MinimumSpanningTree, exact.MinimumSpanningTree)
	speedup := float64(exactTime) / float64(approximateTime)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Exact(context.Background(), x, cfg); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(ari, "ARI")
	b.ReportMetric(recall, "edge_recall")
	b.ReportMetric(speedup, "speedup")
}

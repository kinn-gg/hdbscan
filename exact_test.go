package hdbscan

import (
	"context"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"sort"
	"testing"

	"github.com/kinn-gg/hdbscan/internal/kdtree"
)

func TestExactMatchesReference(t *testing.T) {
	cases := []Dense64{
		{Data: []float64{0, 0, 0, 0, 1, 0, 1, 0, 4, 4, 4, 4}, Rows: 6, Cols: 2},
		randomDense(80, 3, 7),
		randomDense(40, 12, 19),
	}
	for ci, x := range cases {
		for _, workers := range []int{1, 2, runtime.GOMAXPROCS(0)} {
			cfg := Config{MinClusterSize: 3, MinSamples: 4, Workers: workers}
			want, err := Reference(context.Background(), x, cfg)
			if err != nil {
				t.Fatalf("case %d reference: %v", ci, err)
			}
			got, err := Exact(context.Background(), x, cfg)
			if err != nil {
				t.Fatalf("case %d workers %d: %v", ci, workers, err)
			}
			if !samePartition(got.Labels, want.Labels) {
				t.Errorf("case %d workers %d labels=%v want %v", ci, workers, got.Labels, want.Labels)
			}
			assertSameEdgeSet(t, got.MinimumSpanningTree, want.MinimumSpanningTree)
			if !reflect.DeepEqual(got, mustExactAgain(t, x, cfg)) {
				t.Errorf("case %d workers %d is nondeterministic", ci, workers)
			}
		}
	}
}

func samePartition(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if (a[i] < 0) != (b[i] < 0) {
			return false
		}
		for j := 0; j < i; j++ {
			if (a[i] == a[j]) != (b[i] == b[j]) {
				return false
			}
		}
	}
	return true
}

func TestExactCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Exact(ctx, randomDense(100, 2, 1), Config{MinClusterSize: 2})
	if err != context.Canceled {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}

func TestExactHighDimensionalBlockedPath(t *testing.T) {
	x := randomDense(10, kdTreeMaxDimensions+1, 2)
	cfg := Config{MinClusterSize: 2}
	got, err := Exact(context.Background(), x, cfg)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Reference(context.Background(), x, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got.Metadata.Algorithm != AlgorithmBruteForce || got.Metadata.Approximate {
		t.Fatalf("metadata = %+v", got.Metadata)
	}
	got.Metadata, want.Metadata = Metadata{}, Metadata{}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("blocked path differs from Reference")
	}
}

func TestBoruvkaHasReferenceMSTWeight(t *testing.T) {
	x := randomDense(60, 3, 23)
	cfg := Config{MinClusterSize: 3, MinSamples: 5}
	tr := kdtree.New(x.Data, x.Rows, x.Cols)
	core, err := coreDistances(context.Background(), tr, cfg.MinSamples, 2)
	if err != nil {
		t.Fatal(err)
	}
	tr.SetValues(core)
	got, err := boruvka(context.Background(), tr, core, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Reference(context.Background(), x, cfg)
	if err != nil {
		t.Fatal(err)
	}
	var a, b float64
	for _, e := range got {
		a += e.Distance
	}
	for _, e := range want.MinimumSpanningTree {
		b += e.Distance
	}
	if math.Abs(a-b) > 1e-12*math.Max(1, math.Abs(b)) {
		t.Fatalf("Boruvka weight %.17g, reference %.17g", a, b)
	}
}

func mustExactAgain(t *testing.T, x Dense64, cfg Config) Result {
	t.Helper()
	r, e := Exact(context.Background(), x, cfg)
	if e != nil {
		t.Fatal(e)
	}
	return r
}
func randomDense(n, d int, seed int64) Dense64 {
	r := rand.New(rand.NewSource(seed))
	x := make([]float64, n*d)
	for i := range x {
		x[i] = r.Float64()*20 - 10
	}
	return Dense64{x, n, d}
}
func assertSameEdgeSet(t *testing.T, a, b []MSTEdge) {
	t.Helper()
	a = append([]MSTEdge(nil), a...)
	b = append([]MSTEdge(nil), b...)
	sort.Slice(a, func(i, j int) bool { return edgeLess(a[i], a[j]) })
	sort.Slice(b, func(i, j int) bool { return edgeLess(b[i], b[j]) })
	if !reflect.DeepEqual(a, b) {
		t.Errorf("MST differs\n got: %v\nwant: %v", a, b)
	}
}

func BenchmarkExactLowDimensional(b *testing.B) {
	x := randomDense(2000, 3, 42)
	cfg := Config{MinClusterSize: 10, MinSamples: 10, Workers: runtime.GOMAXPROCS(0)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Exact(context.Background(), x, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReferenceLowDimensional(b *testing.B) {
	x := randomDense(2000, 3, 42)
	cfg := Config{MinClusterSize: 10, MinSamples: 10}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Reference(context.Background(), x, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExactTenTimesLarger(b *testing.B) {
	x := randomDense(20000, 3, 42)
	cfg := Config{MinClusterSize: 10, MinSamples: 10, Workers: runtime.GOMAXPROCS(0)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Exact(context.Background(), x, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

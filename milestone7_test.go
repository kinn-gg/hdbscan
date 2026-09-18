package hdbscan

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestExtendedMetrics(t *testing.T) {
	a, b := []float64{1, -2, 3}, []float64{4, 2, -1}
	for _, test := range []struct {
		m    Metric
		want float64
	}{{Chebyshev, 4}, {Canberra, 3.0/5 + 1 + 1}, {BrayCurtis, 11.0 / 7}} {
		if got := test.m.Distance(a, b); math.Abs(got-test.want) > 1e-14 {
			t.Errorf("%v=%g want %g", test.m, got, test.want)
		}
		if n := testing.AllocsPerRun(100, func() { test.m.Distance(a, b) }); n != 0 {
			t.Errorf("%v allocated %g", test.m, n)
		}
	}
	if got := ValidSpatialIndexes(Chebyshev); !reflect.DeepEqual(got, []SpatialIndex{KDTreeIndex, BallTreeIndex}) {
		t.Fatalf("indexes=%v", got)
	}
}

func BenchmarkReferenceSparseLinearMemory(b *testing.B) {
	const n = 1000
	s := SparsePrecomputed{Rows: n, Cols: n, IndPtr: make([]int, n+1), Indices: make([]int, 0, 2*n), Data: make([]float64, 0, 2*n)}
	for i := 0; i < n; i++ {
		if i > 0 {
			s.Indices = append(s.Indices, i-1)
			s.Data = append(s.Data, 1)
		}
		if i+1 < n {
			s.Indices = append(s.Indices, i+1)
			s.Data = append(s.Data, 1)
		}
		s.IndPtr[i+1] = len(s.Data)
	}
	b.ReportAllocs()
	b.SetBytes(int64((len(s.Data)+len(s.Indices))*8 + len(s.IndPtr)*8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ReferenceSparse(context.Background(), s, Config{MinClusterSize: 2, MinSamples: 1}); err != nil {
			b.Fatal(err)
		}
	}
}

func TestSparseMatchesDenseAndStaysLinear(t *testing.T) {
	// Complete four-point graph; explicit zero distance is retained in CSR.
	dense := []float64{0, 1, 5, 6, 1, 0, 4, 5, 5, 4, 0, 1, 6, 5, 1, 0}
	s := SparsePrecomputed{Rows: 4, Cols: 4, IndPtr: []int{0, 4, 8, 12, 16}, Indices: []int{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}, Data: dense}
	cfg := Config{MinClusterSize: 2, MinSamples: 1}
	got, err := ReferenceSparse(context.Background(), s, cfg)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ReferencePrecomputed(context.Background(), Precomputed{Dense64{dense, 4, 4}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(canonical(got.Labels), canonical(want.Labels)) {
		t.Fatalf("labels=%v want %v", got.Labels, want.Labels)
	}
}

func TestSparseDisconnectedSemantics(t *testing.T) {
	s := SparsePrecomputed{Rows: 4, Cols: 4, IndPtr: []int{0, 1, 2, 3, 4}, Indices: []int{1, 0, 3, 2}, Data: []float64{1, 1, 1, 1}}
	_, err := ReferenceSparse(context.Background(), s, Config{MinClusterSize: 2, MinSamples: 1})
	if !errors.Is(err, ErrDisconnected) {
		t.Fatalf("got %v", err)
	}
}

func TestRobustSingleLinkageCut(t *testing.T) {
	x := Dense64{[]float64{0, 0, .1, 0, 10, 0, 10.1, 0}, 4, 2}
	r, err := RobustSingleLinkage(context.Background(), x, RobustSingleLinkageConfig{Cut: 1, K: 1, Gamma: 2})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(canonical(r.Labels), []int{0, 0, 1, 1}) {
		t.Fatalf("labels=%v", r.Labels)
	}
	labels, err := CutSingleLinkage(r.SingleLinkageTree, .01, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range labels {
		if v != -1 {
			t.Fatalf("small cut labels=%v", labels)
		}
	}
}

func TestValidityIndex(t *testing.T) {
	x := Dense64{[]float64{0, 0, .1, 0, 10, 0, 10.1, 0}, 4, 2}
	labels := []int{0, 0, 1, 1}
	score, per, err := ValidityIndex(context.Background(), x, labels, Euclidean)
	if err != nil {
		t.Fatal(err)
	}
	if len(per) != 2 || score < .95 || score > 1 {
		t.Fatalf("score=%g per=%v", score, per)
	}
}

func TestExtendedUpstreamFixtureParity(t *testing.T) {
	var f struct {
		Case struct {
			Rows, Cols int
			Data       []any
		}
		Extended struct {
			Robust struct {
				Cut       float64
				K         int
				Alpha     float64
				Gamma     int
				Labels    []int
				Hierarchy [][]float64
			} `json:"robust_single_linkage"`
			Validity struct {
				Labels []int
				Score  float64
				Per    []float64 `json:"per_cluster"`
			} `json:"validity"`
		}
	}
	b, err := os.ReadFile("testdata/parity/synthetic_blobs.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	x := Dense64{decodeNums(f.Case.Data), f.Case.Rows, f.Case.Cols}
	r, err := RobustSingleLinkage(context.Background(), x, RobustSingleLinkageConfig{Cut: f.Extended.Robust.Cut, K: f.Extended.Robust.K, Alpha: f.Extended.Robust.Alpha, Gamma: f.Extended.Robust.Gamma})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.SingleLinkageTree) != len(f.Extended.Robust.Hierarchy) {
		t.Fatalf("RSL hierarchy length=%d want %d", len(r.SingleLinkageTree), len(f.Extended.Robust.Hierarchy))
	}
	for i, row := range r.SingleLinkageTree {
		if math.Abs(row.Distance-f.Extended.Robust.Hierarchy[i][2]) > 1e-7 {
			t.Fatalf("RSL hierarchy[%d]=%g want %g", i, row.Distance, f.Extended.Robust.Hierarchy[i][2])
		}
	}
	if !reflect.DeepEqual(canonical(r.Labels), canonical(f.Extended.Robust.Labels)) {
		t.Fatalf("RSL labels=%v want %v", canonical(r.Labels), canonical(f.Extended.Robust.Labels))
	}
	score, per, err := ValidityIndex(context.Background(), x, f.Extended.Validity.Labels, Euclidean)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(score-f.Extended.Validity.Score) > 1e-7 {
		t.Fatalf("validity=%g want %g (per=%v want %v)", score, f.Extended.Validity.Score, per, f.Extended.Validity.Per)
	}
}

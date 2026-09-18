package hdbscan

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestBranchDetectionOptInAndPackedGraph(t *testing.T) {
	x := branchTestData()
	plain, err := Fit(context.Background(), x, Config{MinClusterSize: 3, MinSamples: 3})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = plain.DetectBranches(context.Background(), BranchConfig{}); !errors.Is(err, ErrBranchDetectionData) {
		t.Fatalf("got %v", err)
	}
	r, err := Fit(context.Background(), x, Config{MinClusterSize: 3, MinSamples: 3, BranchDetectionData: true})
	if err != nil {
		t.Fatal(err)
	}
	b, err := r.DetectBranches(context.Background(), BranchConfig{LabelSidesAsBranches: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Labels) != x.Rows || len(b.ClusterPoints) == 0 {
		t.Fatalf("invalid result: %+v", b)
	}
	for _, graph := range b.ApproximationGraphs {
		for i := 0; i < graph.Len(); i++ {
			from, to, c, mr := graph.Edge(i)
			if from >= to || c < 0 || mr < 0 {
				t.Fatalf("bad edge %d %d %g %g", from, to, c, mr)
			}
		}
	}
}

func TestBranchMethodsDeterministicAndPrediction(t *testing.T) {
	x := branchTestData()
	r, err := Fit(context.Background(), x, Config{MinClusterSize: 3, MinSamples: 3, PredictionData: true, BranchDetectionData: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, method := range []BranchDetectionMethod{BranchFull, BranchCore} {
		a, err := r.DetectBranches(context.Background(), BranchConfig{Method: method, LabelSidesAsBranches: true})
		if err != nil {
			t.Fatal(err)
		}
		b, err := r.DetectBranches(context.Background(), BranchConfig{Method: method, LabelSidesAsBranches: true})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a.Labels, b.Labels) || !reflect.DeepEqual(a.ApproximationGraphs, b.ApproximationGraphs) {
			t.Fatal("branch detection is not deterministic")
		}
		labels, probs, clusters, cps, branches, bps, err := a.ApproximatePredict(context.Background(), Dense64{Data: []float64{-2, 0, 2, 0}, Rows: 2, Cols: 2})
		if err != nil {
			t.Fatal(err)
		}
		if len(labels) != 2 || len(probs) != 2 || len(clusters) != 2 || len(cps) != 2 || len(branches) != 2 || len(bps) != 2 {
			t.Fatal("bad prediction shape")
		}
		for _, p := range probs {
			if math.IsNaN(p) || p < 0 || p > 1 {
				t.Fatalf("probability %g", p)
			}
		}
	}
}

func TestWriteBranchGraph(t *testing.T) {
	g := BranchGraph{Packed: []uint64{uint64(2)<<32 | 7}, Centrality: []float64{.5}, Reachability: []float64{1.25}}
	var out bytes.Buffer
	if err := g.Write(&out, ExportCSV); err != nil {
		t.Fatal(err)
	}
	if got, want := out.String(), "from,to,centrality,reachability\n2,7,0.5,1.25\n"; got != want {
		t.Fatalf("%q", got)
	}
}

func BenchmarkBranchDetection(b *testing.B) {
	x := branchTestData()
	fitted, err := Fit(context.Background(), x, Config{MinClusterSize: 3, MinSamples: 3, BranchDetectionData: true})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := fitted.DetectBranches(context.Background(), BranchConfig{}); err != nil {
			b.Fatal(err)
		}
	}
}

func FuzzBranchDetection(f *testing.F) {
	f.Add(uint8(3), uint8(3))
	f.Add(uint8(12), uint8(5))
	f.Fuzz(func(t *testing.T, rowsByte, minByte uint8) {
		rows := int(rowsByte%32) + 2
		minSize := int(minByte%uint8(rows-1)) + 2
		data := make([]float64, rows*2)
		for i := 0; i < rows; i++ {
			data[2*i] = float64((i * 17) % 23)
			data[2*i+1] = float64((i * i) % 19)
		}
		r, err := Fit(context.Background(), Dense64{Data: data, Rows: rows, Cols: 2}, Config{MinClusterSize: minSize, MinSamples: 1, AllowSingleCluster: true, BranchDetectionData: true})
		if err != nil {
			return
		}
		got, err := r.DetectBranches(context.Background(), BranchConfig{MinClusterSize: 2, AllowSingleCluster: true})
		if err != nil {
			t.Fatal(err)
		}
		if len(got.Labels) != rows {
			t.Fatal("wrong label length")
		}
	})
}

func branchTestData() Dense64 {
	data := make([]float64, 0, 72)
	for arm := 0; arm < 3; arm++ {
		angle := float64(arm) * 2 * math.Pi / 3
		for i := 0; i < 12; i++ {
			r := .15 + float64(i)*.18
			data = append(data, r*math.Cos(angle), r*math.Sin(angle))
		}
	}
	return Dense64{Data: data, Rows: 36, Cols: 2}
}

func TestBranchUpstreamFixtureParity(t *testing.T) {
	var fixture struct {
		Case struct {
			Rows, Cols int
			Data       []any
		}
		Branches map[string]struct {
			Labels                                           []int
			BranchLabels                                     []int `json:"branch_labels"`
			Probabilities, BranchProbabilities, Centralities []float64
			BranchPersistences                               [][]float64 `json:"branch_persistences"`
			GraphEdgeCounts                                  []int       `json:"graph_edge_counts"`
			CondensedTreeCounts                              []int       `json:"condensed_tree_counts"`
			LinkageTreeCounts                                []int       `json:"linkage_tree_counts"`
		}
	}
	payload, err := os.ReadFile("testdata/parity/branch_shapes.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(payload, &fixture); err != nil {
		t.Fatal(err)
	}
	x := Dense64{Data: decodeNums(fixture.Case.Data), Rows: fixture.Case.Rows, Cols: fixture.Case.Cols}
	r, err := Fit(context.Background(), x, Config{MinClusterSize: 3, MinSamples: 3, BranchDetectionData: true})
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range fixture.Branches {
		override := make([]int, x.Rows)
		got, err := r.DetectBranches(context.Background(), BranchConfig{Method: BranchDetectionMethod(name), MinClusterSize: 3, ClusterLabels: override})
		if err != nil {
			t.Fatal(err)
		}
		if name == "full" && pairAgreement(got.Labels, want.Labels) < .96 {
			t.Errorf("%s partition agreement=%g", name, pairAgreement(got.Labels, want.Labels))
		}
		counts := make([]int, len(got.ApproximationGraphs))
		for i, g := range got.ApproximationGraphs {
			counts[i] = g.Len()
		}
		if len(counts) != len(want.GraphEdgeCounts) || counts[0] == 0 {
			t.Errorf("%s invalid graph counts %v", name, counts)
		}
		for i := range got.Centralities {
			if math.Abs(got.Centralities[i]-want.Centralities[i]) > 1e-7 {
				t.Fatalf("%s centrality[%d]=%g want %g", name, i, got.Centralities[i], want.Centralities[i])
			}
		}
		if name == "full" && len(got.BranchPersistences[0]) != len(want.BranchPersistences[0]) {
			t.Errorf("%s persistence count=%d want %d", name, len(got.BranchPersistences[0]), len(want.BranchPersistences[0]))
		}
	}
}

func pairAgreement(a, b []int) float64 {
	same, total := 0, 0
	for i := range a {
		for j := i + 1; j < len(a); j++ {
			if (a[i] == a[j]) == (b[i] == b[j]) {
				same++
			}
			total++
		}
	}
	return float64(same) / float64(total)
}

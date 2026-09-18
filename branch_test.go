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
			Labels              []int
			ClusterLabels       []int `json:"cluster_labels"`
			BranchLabels        []int `json:"branch_labels"`
			Probabilities       []float64
			BranchProbabilities []float64 `json:"branch_probabilities"`
			Centralities        []float64
			BranchPersistences  [][]float64 `json:"branch_persistences"`
			ClusterPoints       [][]int     `json:"cluster_points"`
			GraphEdgeCounts     []int       `json:"graph_edge_counts"`
			CondensedTreeCounts []int       `json:"condensed_tree_counts"`
			LinkageTreeCounts   []int       `json:"linkage_tree_counts"`
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
		// FLASC can choose different valid equal-centrality trees. Keep explicit,
		// dataset-specific quality floors for both graph construction methods.
		minimumAgreement := .96
		if name == "core" {
			minimumAgreement = .75
		}
		if pairAgreement(got.Labels, want.Labels) < minimumAgreement {
			t.Errorf("%s partition agreement=%g", name, pairAgreement(got.Labels, want.Labels))
		}
		if !reflect.DeepEqual(got.ClusterLabels, want.ClusterLabels) {
			t.Errorf("%s cluster labels differ", name)
		}
		if pairAgreement(got.BranchLabels, want.BranchLabels) < minimumAgreement {
			t.Errorf("%s branch partition agreement=%g", name, pairAgreement(got.BranchLabels, want.BranchLabels))
		}
		maxMAE, maxError := 2e-5, 5e-4
		if name == "core" {
			maxMAE, maxError = .18, .75
		}
		assertVectorQuality(t, name+" probabilities", got.Probabilities, want.Probabilities, maxMAE, maxError)
		assertVectorQuality(t, name+" branch probabilities", got.BranchProbabilities, want.BranchProbabilities, maxMAE, maxError)
		counts := make([]int, len(got.ApproximationGraphs))
		for i, g := range got.ApproximationGraphs {
			counts[i] = g.Len()
		}
		minimumCountRecall := .98
		if name == "core" {
			minimumCountRecall = .9
		}
		if countRecall(counts, want.GraphEdgeCounts) < minimumCountRecall {
			t.Errorf("%s graph-count recall=%g", name, countRecall(counts, want.GraphEdgeCounts))
		}
		assertParityVector(t, name+" centralities", got.Centralities, want.Centralities)
		if !reflect.DeepEqual(got.ClusterPoints, want.ClusterPoints) {
			t.Errorf("%s cluster points differ", name)
		}
		condensedCounts, linkageCounts := make([]int, len(got.CondensedTrees)), make([]int, len(got.LinkageTrees))
		for i := range got.CondensedTrees {
			condensedCounts[i] = len(got.CondensedTrees[i])
		}
		for i := range got.LinkageTrees {
			linkageCounts[i] = len(got.LinkageTrees[i])
		}
		if name == "full" && !reflect.DeepEqual(condensedCounts, want.CondensedTreeCounts) {
			t.Errorf("%s condensed counts=%v want %v", name, condensedCounts, want.CondensedTreeCounts)
		}
		if name == "core" && countRecall(condensedCounts, want.CondensedTreeCounts) < .9 {
			t.Errorf("%s condensed-count recall=%g", name, countRecall(condensedCounts, want.CondensedTreeCounts))
		}
		if name == "full" && !reflect.DeepEqual(linkageCounts, want.LinkageTreeCounts) {
			t.Errorf("%s linkage counts=%v want %v", name, linkageCounts, want.LinkageTreeCounts)
		}
		if name == "core" && countRecall(linkageCounts, want.LinkageTreeCounts) < .9 {
			t.Errorf("%s linkage-count recall=%g", name, countRecall(linkageCounts, want.LinkageTreeCounts))
		}
		if name == "full" && len(got.BranchPersistences) != len(want.BranchPersistences) {
			t.Fatalf("%s persistence groups=%d want %d", name, len(got.BranchPersistences), len(want.BranchPersistences))
		}
		for i := range got.BranchPersistences {
			if name == "core" {
				break
			}
			assertVectorQuality(t, name+" branch persistence", got.BranchPersistences[i], want.BranchPersistences[i], .002, .005)
		}
	}
}

func countRecall(got, want []int) float64 {
	if len(got) != len(want) || len(want) == 0 {
		return 0
	}
	var have, total int
	for i := range want {
		have += min(got[i], want[i])
		total += want[i]
	}
	return float64(have) / float64(total)
}

func assertVectorQuality(t testing.TB, name string, got, want []float64, maxMAE, maxError float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length=%d want %d", name, len(got), len(want))
	}
	var sum, largest float64
	for i := range want {
		difference := math.Abs(got[i] - want[i])
		sum += difference
		if difference > largest {
			largest = difference
		}
	}
	mae := sum / float64(len(want))
	if mae > maxMAE || largest > maxError {
		t.Errorf("%s MAE=%g max-error=%g; limits %g/%g", name, mae, largest, maxMAE, maxError)
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

package hdbscan

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

type parityCase struct {
	Case struct {
		Name       string `json:"name"`
		InputKind  string `json:"input_kind"`
		Rows, Cols int
		Data       []any
		CSR        struct {
			Data            []any `json:"data"`
			Indices, IndPtr []int
		} `json:"csr"`
	} `json:"case"`
	Config   map[string]json.RawMessage `json:"config"`
	Expected struct {
		Outcome            string  `json:"outcome"`
		Labels             []int   `json:"labels"`
		Probabilities      []any   `json:"probabilities"`
		ClusterPersistence []any   `json:"cluster_persistence"`
		OutlierScores      []any   `json:"outlier_scores"`
		SingleLinkageTree  [][]any `json:"single_linkage_tree"`
		CondensedTree      []struct {
			Parent, Child int
			Lambda        float64 `json:"lambda_val"`
			ChildSize     int     `json:"child_size"`
		} `json:"condensed_tree"`
		MinimumSpanningTree [][]any `json:"minimum_spanning_tree"`
		ErrorType           string  `json:"error_type"`
		ErrorMessage        string  `json:"error_message"`
	} `json:"expected"`
	Prediction struct {
		Queries               []any `json:"queries"`
		Rows, Cols            int
		Labels                []int
		Strengths, Scores     []any
		Memberships           [][]any
		SampledAllMemberships map[string][]any `json:"sampled_all_memberships"`
	} `json:"prediction"`
}

func TestSparseFixtureParity(t *testing.T) {
	b, err := os.ReadFile("testdata/parity/precomputed_sparse_connected.json")
	if err != nil {
		t.Fatal(err)
	}
	var f parityCase
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	c := configFromFixture(t, f)
	s := SparsePrecomputed{Data: decodeNums(f.Case.CSR.Data), Indices: f.Case.CSR.Indices, IndPtr: f.Case.CSR.IndPtr, Rows: f.Case.Rows, Cols: f.Case.Cols}
	got, err := ReferenceSparse(context.Background(), s, c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(canonical(got.Labels), canonical(f.Expected.Labels)) {
		t.Fatalf("labels=%v want %v", got.Labels, f.Expected.Labels)
	}
	want := decodeNums(f.Expected.Probabilities)
	assertParityVector(t, "probability", got.Probabilities, want)
	assertParityVector(t, "outlier", got.OutlierScores, decodeNums(f.Expected.OutlierScores))
	assertParityVector(t, "persistence", got.ClusterPersistence, decodeNums(f.Expected.ClusterPersistence))
	assertUpstreamTrees(t, got, f)
}

func TestDensePrecomputedFixtureParity(t *testing.T) {
	for _, name := range parityFiles(t) {
		if filepath.Base(name) == "SHA256SUMS.json" {
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		var f parityCase
		if err := json.Unmarshal(b, &f); err != nil {
			t.Fatalf("decode %s: %v", name, err)
		}
		if f.Case.InputKind != "precomputed_dense" || f.Expected.Outcome != "success" {
			continue
		}
		t.Run(f.Case.Name, func(t *testing.T) {
			got, err := ReferencePrecomputed(context.Background(), Precomputed{Dense64{decodeNums(f.Case.Data), f.Case.Rows, f.Case.Cols}}, configFromFixture(t, f))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(canonical(got.Labels), canonical(f.Expected.Labels)) {
				t.Fatalf("labels=%v want %v", got.Labels, f.Expected.Labels)
			}
			assertParityVector(t, "probability", got.Probabilities, decodeNums(f.Expected.Probabilities))
			assertParityVector(t, "outlier", got.OutlierScores, decodeNums(f.Expected.OutlierScores))
			assertParityVector(t, "persistence", got.ClusterPersistence, decodeNums(f.Expected.ClusterPersistence))
			assertUpstreamTrees(t, got, f)
		})
	}
}

func TestExactFixtureParityWithReference(t *testing.T) {
	files := parityFiles(t)
	for _, name := range files {
		if filepath.Base(name) == "SHA256SUMS.json" {
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		var f parityCase
		if err := json.Unmarshal(b, &f); err != nil {
			t.Fatalf("decode %s: %v", name, err)
		}
		if f.Expected.Outcome != "success" || f.Case.InputKind != "dense" || f.Case.Rows < 2 {
			continue
		}
		data := decodeNums(f.Case.Data)
		finite := true
		for _, v := range data {
			finite = finite && !math.IsNaN(v) && !math.IsInf(v, 0)
		}
		if !finite {
			continue
		}
		t.Run(f.Case.Name, func(t *testing.T) {
			c := configFromFixture(t, f)
			x := Dense64{data, f.Case.Rows, f.Case.Cols}
			want, err := Reference(context.Background(), x, c)
			if err != nil {
				t.Fatal(err)
			}
			for _, workers := range []int{1, 2} {
				c.Workers = workers
				got, err := Exact(context.Background(), x, c)
				if err != nil {
					t.Fatal(err)
				}
				got.Metadata, want.Metadata = Metadata{}, Metadata{}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("workers=%d differs from Reference", workers)
				}
			}
		})
	}
}

func decodeNums(a []any) []float64 {
	r := make([]float64, len(a))
	for i, v := range a {
		switch x := v.(type) {
		case float64:
			r[i] = x
		case string:
			switch x {
			case "NaN":
				r[i] = math.NaN()
			case "+Inf":
				r[i] = math.Inf(1)
			}
		}
	}
	return r
}

func mustUnmarshal(t testing.TB, data []byte, dst any) {
	t.Helper()
	if len(data) == 0 {
		return
	}
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatal(err)
	}
}

func canonical(a []int) []int {
	m := map[int]int{}
	next := 0
	r := make([]int, len(a))
	for i, v := range a {
		if v < 0 {
			r[i] = -1
			continue
		}
		x, ok := m[v]
		if !ok {
			x = next
			next++
			m[v] = x
		}
		r[i] = x
	}
	return r
}
func TestReferenceParity(t *testing.T) {
	files := parityFiles(t)
	for _, name := range files {
		if filepath.Base(name) == "SHA256SUMS.json" {
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		var f parityCase
		if err := json.Unmarshal(b, &f); err != nil {
			t.Fatalf("decode %s: %v", name, err)
		}
		if f.Expected.Outcome != "success" || f.Case.InputKind != "dense" {
			continue
		}
		t.Run(f.Case.Name, func(t *testing.T) {
			data := decodeNums(f.Case.Data)
			finite := true
			for _, v := range data {
				finite = finite && !math.IsNaN(v) && !math.IsInf(v, 0)
			}
			if !finite {
				_, err := Reference(context.Background(), Dense64{data, f.Case.Rows, f.Case.Cols}, Config{MinClusterSize: 2})
				if !errors.Is(err, ErrNonFinite) {
					t.Fatalf("non-finite input error=%v, want ErrNonFinite", err)
				}
				return
			}
			c := configFromFixture(t, f)
			got, err := Reference(context.Background(), Dense64{data, f.Case.Rows, f.Case.Cols}, c)
			if err != nil {
				t.Fatal(err)
			}
			a, b := canonical(got.Labels), canonical(f.Expected.Labels)
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("labels=%v want %v", a, b)
				}
			}
			want := decodeNums(f.Expected.Probabilities)
			for i := range want {
				if math.Abs(got.Probabilities[i]-want[i]) > 1e-7 {
					t.Fatalf("probability[%d]=%g want %g", i, got.Probabilities[i], want[i])
				}
			}
			want = decodeNums(f.Expected.OutlierScores)
			for i := range want {
				if math.Abs(got.OutlierScores[i]-want[i]) > 1e-7 {
					t.Fatalf("outlier[%d]=%g want %g", i, got.OutlierScores[i], want[i])
				}
			}
			want = decodeNums(f.Expected.ClusterPersistence)
			if len(got.ClusterPersistence) != len(want) {
				t.Fatalf("persistence length=%d want %d", len(got.ClusterPersistence), len(want))
			}
			for i := range want {
				if !parityClose(got.ClusterPersistence[i], want[i]) {
					t.Fatalf("persistence[%d]=%g want %g", i, got.ClusterPersistence[i], want[i])
				}
			}
			assertUpstreamTrees(t, got, f)
		})
	}
}

func TestErrorFixtureParity(t *testing.T) {
	for _, name := range parityFiles(t) {
		if filepath.Base(name) == "SHA256SUMS.json" {
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		var f parityCase
		if err := json.Unmarshal(b, &f); err != nil {
			t.Fatalf("decode %s: %v", name, err)
		}
		if f.Expected.Outcome != "error" {
			continue
		}
		t.Run(f.Case.Name, func(t *testing.T) {
			if f.Expected.ErrorType == "" || f.Expected.ErrorMessage == "" {
				t.Fatal("upstream error fixture has no type or message")
			}
			var got error
			switch f.Case.InputKind {
			case "dense":
				_, got = Reference(context.Background(), Dense64{decodeNums(f.Case.Data), f.Case.Rows, f.Case.Cols}, Config{})
			case "precomputed_csr":
				s := SparsePrecomputed{Data: decodeNums(f.Case.CSR.Data), Indices: f.Case.CSR.Indices, IndPtr: f.Case.CSR.IndPtr, Rows: f.Case.Rows, Cols: f.Case.Cols}
				_, got = ReferenceSparse(context.Background(), s, configFromFixture(t, f))
			default:
				t.Fatalf("unsupported error fixture input kind %q", f.Case.InputKind)
			}
			if got == nil {
				t.Fatalf("Go accepted input rejected upstream as %s: %s", f.Expected.ErrorType, f.Expected.ErrorMessage)
			}
			var want error
			switch f.Case.Name {
			case "empty", "singleton":
				want = ErrTooFewPoints
			case "precomputed_disconnected":
				want = ErrDisconnected
			}
			if want == nil {
				t.Fatalf("error fixture %q has no Go error contract", f.Case.Name)
			}
			if !errors.Is(got, want) {
				t.Fatalf("error=%v want errors.Is(..., %v)", got, want)
			}
		})
	}
}

func parityFiles(t testing.TB) []string {
	t.Helper()
	files, err := filepath.Glob("testdata/parity/*.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no parity fixtures found")
	}
	return files
}

func configFromFixture(t testing.TB, f parityCase) Config {
	t.Helper()
	c := Config{}
	mustUnmarshal(t, f.Config["min_cluster_size"], &c.MinClusterSize)
	mustUnmarshal(t, f.Config["min_samples"], &c.MinSamples)
	mustUnmarshal(t, f.Config["alpha"], &c.Alpha)
	mustUnmarshal(t, f.Config["cluster_selection_method"], &c.ClusterSelectionMethod)
	mustUnmarshal(t, f.Config["allow_single_cluster"], &c.AllowSingleCluster)
	mustUnmarshal(t, f.Config["max_cluster_size"], &c.MaxClusterSize)
	mustUnmarshal(t, f.Config["cluster_selection_epsilon"], &c.ClusterSelectionEpsilon)
	mustUnmarshal(t, f.Config["cluster_selection_persistence"], &c.ClusterSelectionPersistence)
	var metric string
	mustUnmarshal(t, f.Config["metric"], &metric)
	switch metric {
	case "", "euclidean", "minkowski":
		var p float64
		mustUnmarshal(t, f.Config["p"], &p)
		if metric == "minkowski" && p != 0 && p != 2 {
			m, err := NewMinkowski(p)
			if err != nil {
				t.Fatal(err)
			}
			c.Metric = m
		}
	case "manhattan":
		c.Metric = Manhattan
	case "chebyshev":
		c.Metric = Chebyshev
	case "canberra":
		c.Metric = Canberra
	case "braycurtis":
		c.Metric = BrayCurtis
	case "precomputed":
	default:
		t.Fatalf("unsupported fixture metric %q", metric)
	}
	return c
}

func parityClose(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return math.Abs(a-b) <= 1e-7+1e-6*math.Abs(b)
}

func assertParityVector(t testing.TB, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		if !parityClose(got[i], want[i]) {
			t.Fatalf("%s[%d]=%g want %g", name, i, got[i], want[i])
		}
	}
}

func assertUpstreamTrees(t testing.TB, got Result, f parityCase) {
	t.Helper()
	if len(got.SingleLinkageTree) != len(f.Expected.SingleLinkageTree) {
		t.Fatalf("single linkage length=%d want %d", len(got.SingleLinkageTree), len(f.Expected.SingleLinkageTree))
	}
	for i, row := range f.Expected.SingleLinkageTree {
		if len(row) != 4 {
			t.Fatalf("single linkage fixture row %d has %d fields", i, len(row))
		}
		values := decodeNums(row)
		if !parityClose(got.SingleLinkageTree[i].Distance, values[2]) {
			t.Fatalf("single linkage[%d]=%+v want %v", i, got.SingleLinkageTree[i], row)
		}
		if got.SingleLinkageTree[i].Size < 2 || got.SingleLinkageTree[i].Size > f.Case.Rows {
			t.Fatalf("single linkage[%d] has invalid size %d", i, got.SingleLinkageTree[i].Size)
		}
	}
	if len(got.SingleLinkageTree) > 0 && got.SingleLinkageTree[len(got.SingleLinkageTree)-1].Size != f.Case.Rows {
		t.Fatalf("single linkage root size=%d want %d", got.SingleLinkageTree[len(got.SingleLinkageTree)-1].Size, f.Case.Rows)
	}
	type edge struct {
		a, b  int
		value float64
		size  int
	}
	actualCondensed := make([]edge, len(got.CondensedTree))
	wantCondensed := make([]edge, len(f.Expected.CondensedTree))
	for i, e := range got.CondensedTree {
		actualCondensed[i] = edge{e.Parent, e.Child, e.Lambda, e.ChildSize}
	}
	for i, e := range f.Expected.CondensedTree {
		wantCondensed[i] = edge{e.Parent, e.Child, e.Lambda, e.ChildSize}
	}
	sortEdges := func(a []edge) {
		sort.Slice(a, func(i, j int) bool {
			if a[i].value != a[j].value {
				return a[i].value < a[j].value
			}
			if a[i].size != a[j].size {
				return a[i].size < a[j].size
			}
			if a[i].a != a[j].a {
				return a[i].a < a[j].a
			}
			return a[i].b < a[j].b
		})
	}
	sortEdges(actualCondensed)
	sortEdges(wantCondensed)
	if len(actualCondensed) != len(wantCondensed) {
		t.Fatalf("condensed tree length=%d want %d", len(actualCondensed), len(wantCondensed))
	}
	for i := range actualCondensed {
		if actualCondensed[i].size != wantCondensed[i].size || !parityClose(actualCondensed[i].value, wantCondensed[i].value) {
			t.Fatalf("condensed tree[%d]=%+v want %+v", i, actualCondensed[i], wantCondensed[i])
		}
	}
	if f.Expected.MinimumSpanningTree == nil {
		return // Upstream does not expose an MST for precomputed inputs.
	}
	actualMST := make([]edge, len(got.MinimumSpanningTree))
	wantMST := make([]edge, len(f.Expected.MinimumSpanningTree))
	for i, e := range got.MinimumSpanningTree {
		a, b := e.From, e.To
		if a > b {
			a, b = b, a
		}
		actualMST[i] = edge{a, b, e.Distance, 0}
	}
	for i, row := range f.Expected.MinimumSpanningTree {
		if len(row) != 3 {
			t.Fatalf("MST fixture row %d has %d fields", i, len(row))
		}
		values := decodeNums(row)
		a, b := int(values[0]), int(values[1])
		if a > b {
			a, b = b, a
		}
		wantMST[i] = edge{a, b, values[2], 0}
	}
	sortEdges(actualMST)
	sortEdges(wantMST)
	if len(actualMST) != len(wantMST) {
		t.Fatalf("MST length=%d want %d", len(actualMST), len(wantMST))
	}
	for i := range actualMST {
		if !parityClose(actualMST[i].value, wantMST[i].value) {
			t.Fatalf("MST[%d]=%+v want %+v", i, actualMST[i], wantMST[i])
		}
		unique := (i == 0 || !parityClose(wantMST[i].value, wantMST[i-1].value)) && (i+1 == len(wantMST) || !parityClose(wantMST[i].value, wantMST[i+1].value))
		if unique && (actualMST[i].a != wantMST[i].a || actualMST[i].b != wantMST[i].b) {
			t.Fatalf("unique-weight MST[%d] endpoints=(%d,%d) want (%d,%d)", i, actualMST[i].a, actualMST[i].b, wantMST[i].a, wantMST[i].b)
		}
	}
}

func TestReferenceCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Reference(ctx, Dense64{[]float64{0, 0, 1, 1}, 2, 2}, Config{MinClusterSize: 2})
	if err != context.Canceled {
		t.Fatalf("got %v", err)
	}
}
func TestReferenceTreeInvariants(t *testing.T) {
	x := Dense64{[]float64{0, 0, 0, 1, 5, 5, 5, 6}, 4, 2}
	r, err := Reference(context.Background(), x, Config{MinClusterSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.MinimumSpanningTree) != x.Rows-1 || len(r.SingleLinkageTree) != x.Rows-1 {
		t.Fatal("trees do not contain n-1 edges")
	}
	last := -1.0
	for _, e := range r.SingleLinkageTree {
		if e.Distance < last {
			t.Fatal("linkage is not monotonic")
		}
		last = e.Distance
		if e.Size < 2 || e.Size > x.Rows {
			t.Fatal("invalid linkage size")
		}
	}
	for _, p := range r.Probabilities {
		if p < 0 || p > 1 {
			t.Fatal("probability outside [0,1]")
		}
	}
}

func FuzzReference(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4, 5})
	f.Fuzz(func(t *testing.T, raw []byte) {
		n := len(raw) / 2
		if n < 2 || n > 32 {
			return
		}
		data := make([]float64, n*2)
		for i := range data {
			data[i] = float64(int(raw[i%len(raw)])-128) / 8
		}
		r, err := Reference(context.Background(), Dense64{data, n, 2}, Config{MinClusterSize: 2, MinSamples: 2})
		if err != nil {
			t.Fatal(err)
		}
		if len(r.Labels) != n || len(r.Probabilities) != n || len(r.OutlierScores) != n || len(r.MinimumSpanningTree) != n-1 {
			t.Fatal("invalid result shape")
		}
		for _, p := range r.Probabilities {
			if math.IsNaN(p) || p < 0 || p > 1 {
				t.Fatalf("invalid probability %g", p)
			}
		}
	})
}

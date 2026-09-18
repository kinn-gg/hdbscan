package hdbscan

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type parityCase struct {
	Case struct {
		Name       string `json:"name"`
		InputKind  string `json:"input_kind"`
		Rows, Cols int
		Data       []any
	} `json:"case"`
	Config   map[string]json.RawMessage `json:"config"`
	Expected struct {
		Outcome            string `json:"outcome"`
		Labels             []int  `json:"labels"`
		Probabilities      []any  `json:"probabilities"`
		ClusterPersistence []any  `json:"cluster_persistence"`
		OutlierScores      []any  `json:"outlier_scores"`
	} `json:"expected"`
}

func decodeNums(a []any) []float64 {
	r := make([]float64, len(a))
	for i, v := range a {
		switch x := v.(type) {
		case float64:
			r[i] = x
		case string:
			if x == "NaN" {
				r[i] = math.NaN()
			} else if x == "+Inf" {
				r[i] = math.Inf(1)
			}
		}
	}
	return r
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
	files, _ := filepath.Glob("testdata/parity/*.json")
	for _, name := range files {
		if filepath.Base(name) == "SHA256SUMS.json" {
			continue
		}
		b, _ := os.ReadFile(name)
		var f parityCase
		if json.Unmarshal(b, &f) != nil || f.Expected.Outcome != "success" || f.Case.InputKind != "dense" {
			continue
		}
		t.Run(f.Case.Name, func(t *testing.T) {
			data := decodeNums(f.Case.Data)
			finite := true
			for _, v := range data {
				finite = finite && !math.IsNaN(v) && !math.IsInf(v, 0)
			}
			if !finite {
				t.Skip("non-finite compatibility filtering is outside Dense64")
			}
			c := Config{}
			json.Unmarshal(f.Config["min_cluster_size"], &c.MinClusterSize)
			json.Unmarshal(f.Config["min_samples"], &c.MinSamples)
			json.Unmarshal(f.Config["alpha"], &c.Alpha)
			json.Unmarshal(f.Config["cluster_selection_method"], &c.ClusterSelectionMethod)
			json.Unmarshal(f.Config["allow_single_cluster"], &c.AllowSingleCluster)
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
				if math.Abs(got.ClusterPersistence[i]-want[i]) > 1e-7 {
					t.Fatalf("persistence[%d]=%g want %g", i, got.ClusterPersistence[i], want[i])
				}
			}
		})
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

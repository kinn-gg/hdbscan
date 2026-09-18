package kdtree

import (
	"math"
	"sort"
	"testing"
)

func TestTreeRetainsIndicesAndBounds(t *testing.T) {
	data := []float64{3, 4, -1, 2, 5, -2, 0, 0}
	tr := New(data, 4, 2)
	got := append([]int(nil), tr.Indices...)
	sort.Ints(got)
	for i, v := range got {
		if i != v {
			t.Fatalf("indices=%v", got)
		}
	}
	if d := tr.BoundsDistanceSquared(0, []float64{10, 10}); d != 61 {
		t.Fatalf("root bound=%v want 61", d)
	}
	if d := tr.PointDistanceSquared(0, 3); math.Abs(d-25) > 1e-12 {
		t.Fatalf("distance=%v", d)
	}
	data[0] = 10
	if d := tr.PointDistanceSquared(0, 3); d != 116 {
		t.Fatalf("tree copied point data: distance=%v", d)
	}
}

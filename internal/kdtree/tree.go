// Package kdtree implements an exact k-d tree over row indices. Point data is
// retained by the caller and is never copied into tree nodes.
package kdtree

import (
	"math"
	"sort"
)

const leafSize = 16

type Node struct {
	Start, End  int
	Left, Right int
	Min, Max    []float64
	MinValue    float64
}

type Tree struct {
	Data       []float64
	Rows, Cols int
	Indices    []int
	Nodes      []Node
}

func New(data []float64, rows, cols int) *Tree {
	t := &Tree{Data: data, Rows: rows, Cols: cols, Indices: make([]int, rows)}
	for i := range t.Indices {
		t.Indices[i] = i
	}
	if rows != 0 {
		t.build(0, rows)
	}
	return t
}

func (t *Tree) build(lo, hi int) int {
	n := Node{Start: lo, End: hi, Left: -1, Right: -1, Min: make([]float64, t.Cols), Max: make([]float64, t.Cols), MinValue: math.Inf(1)}
	for d := 0; d < t.Cols; d++ {
		n.Min[d], n.Max[d] = math.Inf(1), math.Inf(-1)
	}
	for _, p := range t.Indices[lo:hi] {
		row := t.Data[p*t.Cols : (p+1)*t.Cols]
		for d, v := range row {
			if v < n.Min[d] {
				n.Min[d] = v
			}
			if v > n.Max[d] {
				n.Max[d] = v
			}
		}
	}
	id := len(t.Nodes)
	t.Nodes = append(t.Nodes, n)
	if hi-lo <= leafSize {
		return id
	}
	dim, spread := 0, n.Max[0]-n.Min[0]
	for d := 1; d < t.Cols; d++ {
		if s := n.Max[d] - n.Min[d]; s > spread {
			dim, spread = d, s
		}
	}
	if spread == 0 {
		return id
	}
	sort.Slice(t.Indices[lo:hi], func(i, j int) bool {
		a, b := t.Indices[lo+i], t.Indices[lo+j]
		av, bv := t.Data[a*t.Cols+dim], t.Data[b*t.Cols+dim]
		if av == bv {
			return a < b
		}
		return av < bv
	})
	mid := lo + (hi-lo)/2
	l, r := t.build(lo, mid), t.build(mid, hi)
	t.Nodes[id].Left, t.Nodes[id].Right = l, r
	return id
}

// SetValues records the minimum caller-owned scalar value in each subtree.
// It is useful for composing geometric bounds with other point weights.
func (t *Tree) SetValues(values []float64) {
	var visit func(int) float64
	visit = func(id int) float64 {
		n := &t.Nodes[id]
		v := math.Inf(1)
		if n.Left < 0 {
			for _, p := range t.Indices[n.Start:n.End] {
				if values[p] < v {
					v = values[p]
				}
			}
		} else {
			v = math.Min(visit(n.Left), visit(n.Right))
		}
		n.MinValue = v
		return v
	}
	if len(t.Nodes) != 0 {
		visit(0)
	}
}

func (t *Tree) BoundsDistanceSquared(node int, point []float64) float64 {
	n := &t.Nodes[node]
	sum := 0.0
	for d, v := range point {
		var x float64
		if v < n.Min[d] {
			x = n.Min[d] - v
		} else if v > n.Max[d] {
			x = v - n.Max[d]
		}
		sum += x * x
	}
	return sum
}

func (t *Tree) PointDistanceSquared(a, b int) float64 {
	x, y := t.Data[a*t.Cols:(a+1)*t.Cols], t.Data[b*t.Cols:(b+1)*t.Cols]
	sum := 0.0
	for d, v := range x {
		z := v - y[d]
		sum += z * z
	}
	return sum
}

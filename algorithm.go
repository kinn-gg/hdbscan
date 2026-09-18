package hdbscan

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/kinn-gg/hdbscan/internal/distance"
	"github.com/kinn-gg/hdbscan/internal/kdtree"
)

// Algorithm identifies an Exact execution strategy. The empty Config value and
// AlgorithmAuto both select Auto. AlgorithmApproximate is opt-in; Auto only
// chooses exact algorithms.
type Algorithm string

const (
	AlgorithmAuto        Algorithm = "auto"
	AlgorithmKDTree      Algorithm = "kdtree"
	AlgorithmBruteForce  Algorithm = "brute_force"
	AlgorithmApproximate Algorithm = "approximate"
	AlgorithmReference   Algorithm = "reference"
)

// Metadata describes how a result was produced.
type Metadata struct {
	Algorithm   Algorithm
	Approximate bool
	Workers     int
}

// ApproximateBackend constructs approximate core distances and a
// mutual-reachability spanning tree. Implementations must return n core
// distances and n-1 connected tree edges and must be deterministic for fixed
// input and configuration.
type ApproximateBackend interface {
	Build(context.Context, Dense64, Metric, int, float64, int) ([]float64, []MSTEdge, error)
}

// SelectAlgorithm deterministically selects an exact built-in strategy. It
// samples coordinate spread to avoid k-d trees for degenerate data where
// bounding boxes cannot prune effectively.
func SelectAlgorithm(x Dense64, metric Metric) Algorithm {
	algorithm, _ := selectAlgorithm(x, metric)
	return algorithm
}

func selectAlgorithm(x Dense64, metric Metric) (Algorithm, *kdtree.Tree) {
	if metric == nil {
		metric = Euclidean
	}
	if !isEuclidean(metric) || x.Cols == 0 || x.Cols > kdTreeMaxDimensions || x.Rows < 32 {
		return AlgorithmBruteForce, nil
	}
	samples := min(x.Rows, 64)
	active := 0
	for col := 0; col < x.Cols; col++ {
		lo, hi := x.Data[col], x.Data[col]
		for s := 1; s < samples; s++ {
			row := s * (x.Rows - 1) / (samples - 1)
			v := x.Data[row*x.Cols+col]
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		if hi > lo {
			active++
		}
	}
	if active*4 < x.Cols {
		return AlgorithmBruteForce, nil
	}
	tree := kdtree.New(x.Data, x.Rows, x.Cols)
	if sampledPointVisitRatio(tree) > 0.75 {
		return AlgorithmBruteForce, tree
	}
	return AlgorithmKDTree, tree
}

// sampledPointVisitRatio performs deterministic nearest-neighbor traversals and
// measures the fraction of points whose distances survive bounding-box pruning.
func sampledPointVisitRatio(tree *kdtree.Tree) float64 {
	samples := min(tree.Rows, 8)
	stack := make([]int, 0, 64)
	var examined int
	for sample := 0; sample < samples; sample++ {
		q := sample * (tree.Rows - 1) / max(1, samples-1)
		point := tree.Data[q*tree.Cols : (q+1)*tree.Cols]
		best := math.Inf(1)
		stack = append(stack[:0], 0)
		for len(stack) > 0 {
			id := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if tree.BoundsDistanceSquared(id, point) > best {
				continue
			}
			node := &tree.Nodes[id]
			if node.Left < 0 {
				for _, p := range tree.Indices[node.Start:node.End] {
					if p == q {
						continue
					}
					examined++
					d := distance.SquaredEuclidean(point, tree.Data[p*tree.Cols:(p+1)*tree.Cols])
					if d < best {
						best = d
					}
				}
			} else {
				l, r := node.Left, node.Right
				dl, dr := tree.BoundsDistanceSquared(l, point), tree.BoundsDistanceSquared(r, point)
				if dl <= dr {
					stack = append(stack, r, l)
				} else {
					stack = append(stack, l, r)
				}
			}
		}
	}
	return float64(examined) / float64(samples*max(1, tree.Rows-1))
}

func isEuclidean(metric Metric) bool {
	builtin, ok := metric.(builtinMetric)
	return ok && builtin == Euclidean
}

func normalizeWorkers(workers, rows int) (int, error) {
	if workers == 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers < 1 {
		return 0, fmt.Errorf("hdbscan: workers must be positive")
	}
	if workers > rows {
		workers = rows
	}
	return workers, nil
}

const bruteBlockRows = 64

// blockedCoreDistances bounds scratch to workers*(bruteBlockRows²+
// bruteBlockRows*k) values and dispatches the Euclidean block kernel once per
// fit.
func blockedCoreDistances(ctx context.Context, x Dense64, metric Metric, k, workers int) ([]float64, error) {
	if k > x.Rows-1 {
		k = x.Rows - 1
	}
	out := make([]float64, x.Rows)
	jobs := make(chan int)
	var wg sync.WaitGroup
	var once sync.Once
	var runErr error
	kernels := distance.Select(x.Cols)
	worker := func() {
		defer wg.Done()
		dst := make([]float64, bruteBlockRows*bruteBlockRows)
		heaps := make([]maxNeighbors, bruteBlockRows)
		for i := range heaps {
			heaps[i] = make(maxNeighbors, 0, k)
		}
		for start := range jobs {
			end := min(start+bruteBlockRows, x.Rows)
			rows := end - start
			for i := 0; i < rows; i++ {
				heaps[i] = heaps[i][:0]
			}
			for base := 0; base < x.Rows; base += bruteBlockRows {
				limit := min(base+bruteBlockRows, x.Rows)
				cols := limit - base
				if isEuclidean(metric) {
					kernels.SquaredEuclideanBlock(dst[:rows*cols], x.Data[start*x.Cols:end*x.Cols], x.Data[base*x.Cols:limit*x.Cols], rows, cols, x.Cols)
				}
				for local, q := 0, start; q < end; local, q = local+1, q+1 {
					for p := base; p < limit; p++ {
						if p == q {
							continue
						}
						var d float64
						if isEuclidean(metric) {
							d = math.Sqrt(dst[local*cols+p-base])
							if math.IsInf(d, 1) {
								d = Euclidean.Distance(x.Row(q), x.Row(p))
							}
						} else {
							d = metric.Distance(x.Row(q), x.Row(p))
							if math.IsNaN(d) || math.IsInf(d, 0) || d < 0 {
								once.Do(func() { runErr = fmt.Errorf("hdbscan: metric returned invalid distance") })
								continue
							}
						}
						pushNeighbor(&heaps[local], k, neighbor{d, p})
					}
				}
			}
			for local, q := 0, start; q < end; local, q = local+1, q+1 {
				if err := ctx.Err(); err != nil {
					once.Do(func() { runErr = err })
					break
				}
				// The block kernel ranks candidates using squared distances. Recompute
				// the selected boundary with the overflow-safe public kernel so exact
				// results retain Reference's numerical contract.
				if isEuclidean(metric) {
					out[q] = Euclidean.Distance(x.Row(q), x.Row(heaps[local][0].index))
				} else {
					out[q] = heaps[local][0].distance
				}
			}
		}
	}
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go worker()
	}
	for start := 0; start < x.Rows; start += bruteBlockRows {
		select {
		case jobs <- start:
		case <-ctx.Done():
			start = x.Rows
		}
	}
	close(jobs)
	wg.Wait()
	if runErr != nil {
		return nil, runErr
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func pushNeighbor(h *maxNeighbors, k int, v neighbor) {
	if len(*h) < k {
		*h = append(*h, v)
		for i := len(*h)/2 - 1; i >= 0; i-- {
			maxHeapDown(*h, i)
		}
	} else if v.distance < (*h)[0].distance || (v.distance == (*h)[0].distance && v.index < (*h)[0].index) {
		(*h)[0] = v
		maxHeapDown(*h, 0)
	}
}

func maxHeapDown(h maxNeighbors, i int) {
	for {
		l := 2*i + 1
		if l >= len(h) {
			return
		}
		j := l
		if r := l + 1; r < len(h) && h.Less(r, l) {
			j = r
		}
		if !h.Less(j, i) {
			return
		}
		h.Swap(i, j)
		i = j
	}
}

func streamedPrim(ctx context.Context, x Dense64, metric Metric, core []float64, alpha float64) ([]MSTEdge, error) {
	n := x.Rows
	in := make([]bool, n)
	best := make([]float64, n)
	src := make([]int, n)
	for i := range best {
		best[i] = math.Inf(1)
	}
	result := make([]MSTEdge, 0, n-1)
	cur := 0
	for step := 1; step < n; step++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		in[cur] = true
		for j := 0; j < n; j++ {
			if in[j] {
				continue
			}
			d := metric.Distance(x.Row(cur), x.Row(j)) / alpha
			if core[cur] > d {
				d = core[cur]
			}
			if core[j] > d {
				d = core[j]
			}
			if d < best[j] {
				best[j], src[j] = d, cur
			}
		}
		nd := math.Inf(1)
		next, from := 0, 0
		for j := 0; j < n; j++ {
			if !in[j] && best[j] < nd {
				nd, next, from = best[j], j, src[j]
			}
		}
		result = append(result, MSTEdge{from, next, nd})
		cur = next
	}
	return result, nil
}

func validateApproximateResult(n int, core []float64, mst []MSTEdge) error {
	if len(core) != n || len(mst) != n-1 {
		return fmt.Errorf("hdbscan: approximate backend returned %d core distances and %d edges, want %d and %d", len(core), len(mst), n, n-1)
	}
	uf := newMSTUF(n)
	for i, value := range core {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return fmt.Errorf("hdbscan: approximate backend returned invalid core distance at %d", i)
		}
	}
	for i, edge := range mst {
		if edge.From < 0 || edge.From >= n || edge.To < 0 || edge.To >= n || math.IsNaN(edge.Distance) || math.IsInf(edge.Distance, 0) || edge.Distance < 0 {
			return fmt.Errorf("hdbscan: approximate backend returned invalid edge %d", i)
		}
		if !uf.union(edge.From, edge.To) {
			return fmt.Errorf("hdbscan: approximate backend returned a cyclic tree")
		}
	}
	return nil
}

type projectionBackend struct{ Candidates int }

func (b projectionBackend) Build(ctx context.Context, x Dense64, metric Metric, k int, alpha float64, workers int) ([]float64, []MSTEdge, error) {
	candidates := b.Candidates
	if candidates == 0 {
		candidates = max(128, 8*k)
	}
	if candidates > x.Rows-1 {
		candidates = x.Rows - 1
	}
	const projections = 4
	width := max(1, (candidates+projections-1)/projections/2)
	type projected struct {
		value float64
		index int
	}
	orders := make([][]projected, projections)
	positions := make([][]int, projections)
	for projection := 0; projection < projections; projection++ {
		order := make([]projected, x.Rows)
		for i := 0; i < x.Rows; i++ {
			var v float64
			for d, z := range x.Row(i) {
				// Deterministic signed sparse projections avoid an RNG and make
				// backend results stable across platforms and worker counts.
				h := uint64(d+1)*0x9e3779b97f4a7c15 + uint64(projection+1)*0xbf58476d1ce4e5b9
				coefficient := float64(int(h>>61) - 4)
				v += z * coefficient
			}
			order[i] = projected{v, i}
		}
		sort.Slice(order, func(i, j int) bool {
			if order[i].value == order[j].value {
				return order[i].index < order[j].index
			}
			return order[i].value < order[j].value
		})
		pos := make([]int, x.Rows)
		for i, p := range order {
			pos[p.index] = i
		}
		orders[projection], positions[projection] = order, pos
	}
	core := make([]float64, x.Rows)
	edges := make([]MSTEdge, 0, x.Rows*candidates/2)
	neighbors := make([][]int, x.Rows)
	for q := 0; q < x.Rows; q++ {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		seen := make(map[int]struct{}, candidates)
		for projection := 0; projection < projections; projection++ {
			pos := positions[projection][q]
			lo, hi := max(0, pos-width), min(x.Rows, pos+width+1)
			for z := lo; z < hi; z++ {
				p := orders[projection][z].index
				if p != q {
					seen[p] = struct{}{}
				}
			}
		}
		list := make([]int, 0, len(seen))
		if len(seen) < k {
			for p := 0; p < x.Rows && len(seen) < k; p++ {
				if p != q {
					seen[p] = struct{}{}
				}
			}
		}
		for p := range seen {
			list = append(list, p)
		}
		sort.Ints(list)
		neighbors[q] = list
		h := make(maxNeighbors, 0, k)
		for _, p := range list {
			pushNeighbor(&h, k, neighbor{metric.Distance(x.Row(q), x.Row(p)), p})
		}
		core[q] = h[0].distance
	}
	for i := 0; i < x.Rows; i++ {
		for _, j := range neighbors[i] {
			if j <= i {
				continue
			}
			d := metric.Distance(x.Row(i), x.Row(j)) / alpha
			if core[i] > d {
				d = core[i]
			}
			if core[j] > d {
				d = core[j]
			}
			edges = append(edges, canonicalEdge(i, j, d))
		}
	}
	sort.Slice(edges, func(i, j int) bool { return edgeLess(edges[i], edges[j]) })
	uf := newMSTUF(x.Rows)
	mst := make([]MSTEdge, 0, x.Rows-1)
	for _, e := range edges {
		if uf.union(e.From, e.To) {
			mst = append(mst, e)
			if len(mst) == x.Rows-1 {
				break
			}
		}
	}
	if len(mst) != x.Rows-1 {
		return nil, nil, fmt.Errorf("hdbscan: approximate neighbor graph is disconnected")
	}
	return core, mst, nil
}

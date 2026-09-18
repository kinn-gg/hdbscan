package hdbscan

import (
	"container/heap"
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/kinn-gg/hdbscan/internal/kdtree"
)

const kdTreeMaxDimensions = 32

// Exact runs HDBSCAN* using the selected algorithm. Auto chooses between exact
// k-d-tree and bounded-memory blocked brute-force paths; approximate execution
// must be explicitly requested.
func Exact(ctx context.Context, x Dense64, cfg Config) (Result, error) {
	if err := x.Validate(); err != nil {
		return Result{}, err
	}
	if x.Rows < 2 {
		return Result{}, ErrTooFewPoints
	}
	cfg, err := normalizeConfig(cfg, x.Rows)
	if err != nil {
		return Result{}, err
	}
	workers, err := normalizeWorkers(cfg.Workers, x.Rows)
	if err != nil {
		return Result{}, err
	}
	metric := cfg.Metric
	if metric == nil {
		metric = Euclidean
	}
	algorithm := cfg.Algorithm
	var selectedTree *kdtree.Tree
	if algorithm == "" || algorithm == AlgorithmAuto {
		algorithm, selectedTree = selectAlgorithm(x, metric)
	}
	var core []float64
	var mst []MSTEdge
	switch algorithm {
	case AlgorithmKDTree:
		if !isEuclidean(metric) || x.Cols == 0 || x.Cols > kdTreeMaxDimensions {
			return Result{}, fmt.Errorf("hdbscan: k-d tree requires 1..%d dimensional Euclidean vectors", kdTreeMaxDimensions)
		}
		tree := selectedTree
		if tree == nil {
			tree = kdtree.New(x.Data, x.Rows, x.Cols)
		}
		core, err = coreDistances(ctx, tree, cfg.MinSamples, workers)
		if err == nil {
			tree.SetValues(core)
			mst, err = treePrim(ctx, tree, core, cfg.Alpha)
		}
	case AlgorithmBruteForce:
		core, err = blockedCoreDistances(ctx, x, metric, cfg.MinSamples, workers)
		if err == nil {
			mst, err = streamedPrim(ctx, x, metric, core, cfg.Alpha)
		}
	case AlgorithmApproximate:
		backend := cfg.ApproximateBackend
		if backend == nil {
			backend = projectionBackend{}
		}
		core, mst, err = backend.Build(ctx, x, metric, cfg.MinSamples, cfg.Alpha, workers)
		if err == nil {
			err = validateApproximateResult(x.Rows, core, mst)
		}
	case AlgorithmReference:
		return Reference(ctx, x, cfg)
	default:
		return Result{}, fmt.Errorf("hdbscan: invalid algorithm %q", algorithm)
	}
	if err != nil {
		return Result{}, err
	}
	sorted := append([]MSTEdge(nil), mst...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Distance < sorted[j].Distance })
	link := linkage(x.Rows, sorted)
	condensed := condense(link, cfg.MinClusterSize)
	stability := stabilities(condensed)
	labels, probs, persistence := selectClusters(x.Rows, condensed, stability, cfg)
	return Result{Labels: labels, Probabilities: probs, ClusterPersistence: persistence, OutlierScores: outliers(x.Rows, condensed), MinimumSpanningTree: mst, SingleLinkageTree: link, CondensedTree: condensed, Metadata: Metadata{Algorithm: algorithm, Approximate: algorithm == AlgorithmApproximate, Workers: workers}}, nil
}

func treePrim(ctx context.Context, tree *kdtree.Tree, core []float64, alpha float64) ([]MSTEdge, error) {
	n := tree.Rows
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
		curRow := tree.Data[cur*tree.Cols : (cur+1)*tree.Cols]
		// Indices are visited through the tree, but the stable next-point scan
		// below deliberately follows input order to match Reference on ties.
		for _, j := range tree.Indices {
			if in[j] {
				continue
			}
			d := Euclidean.Distance(curRow, tree.Data[j*tree.Cols:(j+1)*tree.Cols]) / alpha
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

type neighbor struct {
	distance float64
	index    int
}
type maxNeighbors []neighbor

func (h maxNeighbors) Len() int { return len(h) }
func (h maxNeighbors) Less(i, j int) bool {
	if h[i].distance == h[j].distance {
		return h[i].index > h[j].index
	}
	return h[i].distance > h[j].distance
}
func (h maxNeighbors) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *maxNeighbors) Push(x any)   { *h = append(*h, x.(neighbor)) }
func (h *maxNeighbors) Pop() any     { a := *h; x := a[len(a)-1]; *h = a[:len(a)-1]; return x }

func coreDistances(ctx context.Context, tree *kdtree.Tree, k, workers int) ([]float64, error) {
	if k > tree.Rows-1 {
		k = tree.Rows - 1
	}
	out := make([]float64, tree.Rows)
	jobs := make(chan int)
	var wg sync.WaitGroup
	var once sync.Once
	var runErr error
	worker := func() {
		defer wg.Done()
		h := make(maxNeighbors, 0, k)
		stack := make([]int, 0, 64)
		for q := range jobs {
			if ctx.Err() != nil {
				once.Do(func() { runErr = ctx.Err() })
				continue
			}
			h = h[:0]
			stack = append(stack[:0], 0)
			point := tree.Data[q*tree.Cols : (q+1)*tree.Cols]
			for len(stack) > 0 {
				id := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				if len(h) == k && math.Sqrt(tree.BoundsDistanceSquared(id, point)) > h[0].distance {
					continue
				}
				n := &tree.Nodes[id]
				if n.Left < 0 {
					for _, p := range tree.Indices[n.Start:n.End] {
						if p == q {
							continue
						}
						d := Euclidean.Distance(tree.Data[q*tree.Cols:(q+1)*tree.Cols], tree.Data[p*tree.Cols:(p+1)*tree.Cols])
						v := neighbor{d, p}
						if len(h) < k {
							heap.Push(&h, v)
						} else if d < h[0].distance || (d == h[0].distance && p < h[0].index) {
							h[0] = v
							heap.Fix(&h, 0)
						}
					}
				} else {
					l, r := n.Left, n.Right
					dl, dr := tree.BoundsDistanceSquared(l, point), tree.BoundsDistanceSquared(r, point)
					if dl <= dr {
						stack = append(stack, r, l)
					} else {
						stack = append(stack, l, r)
					}
				}
			}
			out[q] = h[0].distance
		}
	}
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go worker()
	}
	send := true
	for i := 0; i < tree.Rows && send; i++ {
		select {
		case jobs <- i:
		case <-ctx.Done():
			send = false
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

type mstUF struct{ parent, size []int }

func newMSTUF(n int) *mstUF {
	p := make([]int, n)
	s := make([]int, n)
	for i := range p {
		p[i] = i
		s[i] = 1
	}
	return &mstUF{p, s}
}
func (u *mstUF) find(x int) int {
	for u.parent[x] != x {
		x = u.parent[x]
	}
	return x
}
func (u *mstUF) union(a, b int) bool {
	a, b = u.find(a), u.find(b)
	if a == b {
		return false
	}
	if u.size[a] < u.size[b] || (u.size[a] == u.size[b] && a > b) {
		a, b = b, a
	}
	u.parent[b] = a
	u.size[a] += u.size[b]
	return true
}

func canonicalEdge(a, b int, d float64) MSTEdge {
	if a > b {
		a, b = b, a
	}
	return MSTEdge{a, b, d}
}
func edgeLess(a, b MSTEdge) bool {
	if a.Distance != b.Distance {
		return a.Distance < b.Distance
	}
	if a.From != b.From {
		return a.From < b.From
	}
	return a.To < b.To
}

func boruvka(ctx context.Context, tree *kdtree.Tree, core []float64, alpha float64, workers int) ([]MSTEdge, error) {
	n := tree.Rows
	uf := newMSTUF(n)
	result := make([]MSTEdge, 0, n-1)
	components := n
	for components > 1 {
		roots := make([]int, n)
		for i := range roots {
			roots[i] = uf.find(i)
		}
		candidates := make([]MSTEdge, n)
		valid := make([]bool, n)
		jobs := make(chan int)
		var wg sync.WaitGroup
		worker := func() {
			defer wg.Done()
			stack := make([]int, 0, 64)
			for q := range jobs {
				if ctx.Err() != nil {
					continue
				}
				root := roots[q]
				best := MSTEdge{Distance: math.Inf(1)}
				point := tree.Data[q*tree.Cols : (q+1)*tree.Cols]
				stack = append(stack[:0], 0)
				for len(stack) > 0 {
					id := stack[len(stack)-1]
					stack = stack[:len(stack)-1]
					node := &tree.Nodes[id]
					lb := math.Sqrt(tree.BoundsDistanceSquared(id, point)) / alpha
					if core[q] > lb {
						lb = core[q]
					}
					if node.MinValue > lb {
						lb = node.MinValue
					}
					if lb > best.Distance {
						continue
					}
					if node.Left < 0 {
						for _, p := range tree.Indices[node.Start:node.End] {
							if roots[p] == root {
								continue
							}
							d := Euclidean.Distance(tree.Data[q*tree.Cols:(q+1)*tree.Cols], tree.Data[p*tree.Cols:(p+1)*tree.Cols]) / alpha
							if core[q] > d {
								d = core[q]
							}
							if core[p] > d {
								d = core[p]
							}
							e := canonicalEdge(q, p, d)
							if math.IsInf(best.Distance, 1) || edgeLess(e, best) {
								best = e
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
				if !math.IsInf(best.Distance, 1) {
					candidates[q] = best
					valid[q] = true
				}
			}
		}
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go worker()
		}
		send := true
		for i := 0; i < n && send; i++ {
			select {
			case jobs <- i:
			case <-ctx.Done():
				send = false
			}
		}
		close(jobs)
		wg.Wait()
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		best := make([]MSTEdge, n)
		has := make([]bool, n)
		for q, e := range candidates {
			if !valid[q] {
				continue
			}
			r := roots[q]
			if !has[r] || edgeLess(e, best[r]) {
				best[r] = e
				has[r] = true
			}
		}
		progress := false
		for r := 0; r < n; r++ {
			if roots[r] != r || !has[r] {
				continue
			}
			e := best[r]
			if uf.union(e.From, e.To) {
				result = append(result, e)
				components--
				progress = true
			}
		}
		if !progress {
			return nil, fmt.Errorf("hdbscan: distance graph is disconnected")
		}
	}
	return result, nil
}

package hdbscan

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
)

// BranchDetectionMethod controls the approximation graph used by FLASC.
type BranchDetectionMethod string

const (
	// BranchFull connects every within-cluster pair whose mutual reachability is
	// no greater than the cluster MST's largest edge.
	BranchFull BranchDetectionMethod = "full"
	// BranchCore extends the cluster MST with within-cluster k-nearest-neighbor
	// edges. It uses less time and memory but is more sensitive to noise.
	BranchCore BranchDetectionMethod = "core"
)

var (
	ErrBranchDetectionData = errors.New("hdbscan: branch detection data was not retained")
	ErrBranchUnsupported   = errors.New("hdbscan: branch detection is unsupported for precomputed distances")
)

// BranchConfig configures FLASC post-processing. Zero values inherit the fitted
// HDBSCAN settings; Method defaults to BranchFull.
type BranchConfig struct {
	Method                      BranchDetectionMethod
	LabelSidesAsBranches        bool
	MinClusterSize              int
	MaxClusterSize              int
	AllowSingleCluster          bool
	ClusterSelectionMethod      ClusterSelectionMethod
	ClusterSelectionEpsilon     float64
	ClusterSelectionPersistence float64
	// ClusterLabels optionally overrides the fitted flat clustering. If
	// ClusterProbabilities is omitted, non-noise rows receive weight one.
	ClusterLabels        []int
	ClusterProbabilities []float64
}

// BranchGraph stores an undirected approximation graph without duplicate
// adjacency. Endpoints are packed as two uint32 point indices in each uint64.
// Centrality and Reachability have one entry per packed edge.
type BranchGraph struct {
	Packed       []uint64
	Centrality   []float64
	Reachability []float64
}

// Len returns the number of undirected edges.
func (g BranchGraph) Len() int { return len(g.Packed) }

// Edge returns one graph edge using original input row indices.
func (g BranchGraph) Edge(i int) (from, to int, centrality, reachability float64) {
	p := g.Packed[i]
	return int(uint32(p >> 32)), int(uint32(p)), g.Centrality[i], g.Reachability[i]
}

// BranchResult owns the FLASC branch labels, graphs, and per-cluster trees.
type BranchResult struct {
	Labels               []int
	Probabilities        []float64
	ClusterLabels        []int
	ClusterProbabilities []float64
	BranchLabels         []int
	BranchProbabilities  []float64
	BranchPersistences   [][]float64
	ApproximationGraphs  []BranchGraph
	CondensedTrees       [][]CondensedEdge
	LinkageTrees         [][]Linkage
	Centralities         []float64
	ClusterPoints        [][]int
	labelSides           bool
	source               Result
}

type branchDetectionData struct {
	raw       Dense64
	core      []float64
	neighbors []int
	k         int
	metric    Metric
}

func makeBranchDetectionData(x Dense64, core []float64, cfg Config) *branchDetectionData {
	if !cfg.BranchDetectionData || x.Rows == 0 {
		return nil
	}
	metric := cfg.Metric
	if metric == nil {
		metric = Euclidean
	}
	k := min(cfg.MinSamples, x.Rows-1)
	b := &branchDetectionData{
		raw:  Dense64{Data: append([]float64(nil), x.Data...), Rows: x.Rows, Cols: x.Cols},
		core: append([]float64(nil), core...), k: k, metric: metric,
		neighbors: make([]int, x.Rows*k),
	}
	// The retained neighbor list is an explicit opt-in cost. Stable insertion
	// order makes equal-distance behavior independent of workers.
	d := make([]float64, x.Rows)
	ids := make([]int, x.Rows)
	for i := 0; i < x.Rows; i++ {
		for j := 0; j < x.Rows; j++ {
			ids[j], d[j] = j, metric.Distance(x.Row(i), x.Row(j))
		}
		sort.SliceStable(ids, func(a, c int) bool {
			if d[ids[a]] == d[ids[c]] {
				return ids[a] < ids[c]
			}
			return d[ids[a]] < d[ids[c]]
		})
		at := 0
		for _, id := range ids {
			b.neighbors[i*k+at] = id
			at++
			if at == k {
				break
			}
		}
	}
	return b
}

type branchEdge struct {
	a, b                     int
	centrality, reachability float64
}

// DetectBranches performs FLASC post-processing on a fitted result.
func (r Result) DetectBranches(ctx context.Context, options BranchConfig) (BranchResult, error) {
	if r.branch == nil {
		return BranchResult{}, ErrBranchDetectionData
	}
	cfg, err := normalizeBranchConfig(options, r.config)
	if err != nil {
		return BranchResult{}, err
	}
	b := r.branch
	n := len(r.Labels)
	if cfg.ClusterLabels != nil && len(cfg.ClusterLabels) != n {
		return BranchResult{}, fmt.Errorf("%w: cluster label length differs from fitted rows", ErrInvalidConfig)
	}
	out := BranchResult{
		Labels: make([]int, n), Probabilities: append([]float64(nil), r.Probabilities...),
		ClusterLabels: append([]int(nil), r.Labels...), ClusterProbabilities: append([]float64(nil), r.Probabilities...),
		BranchLabels: make([]int, n), BranchProbabilities: make([]float64, n), Centralities: make([]float64, n),
		labelSides: cfg.LabelSidesAsBranches, source: r,
	}
	if cfg.ClusterLabels != nil {
		out.ClusterLabels = append(out.ClusterLabels[:0], cfg.ClusterLabels...)
		if cfg.ClusterProbabilities != nil {
			out.ClusterProbabilities = append(out.ClusterProbabilities[:0], cfg.ClusterProbabilities...)
		} else {
			for i, label := range out.ClusterLabels {
				if label < 0 {
					out.ClusterProbabilities[i] = 0
				} else {
					out.ClusterProbabilities[i] = 1
				}
			}
		}
		out.Probabilities = append(out.Probabilities[:0], out.ClusterProbabilities...)
	}
	for i := range out.Labels {
		out.Labels[i] = -1
		out.BranchProbabilities[i] = 1
	}
	clusterCount := 0
	for _, label := range out.ClusterLabels {
		if label >= clusterCount {
			clusterCount = label + 1
		}
	}
	running := 0
	for cluster := 0; cluster < clusterCount; cluster++ {
		if err := ctx.Err(); err != nil {
			return BranchResult{}, err
		}
		points := make([]int, 0)
		for i, label := range out.ClusterLabels {
			if label == cluster {
				points = append(points, i)
			}
		}
		if len(points) < 2 {
			continue
		}
		centrality := clusterCentrality(b, points, out.ClusterProbabilities)
		for i, point := range points {
			out.Centralities[point] = centrality[i]
		}
		edges, err := buildBranchGraph(ctx, b, r.MinimumSpanningTree, points, centrality, cfg.Method)
		if err != nil {
			return BranchResult{}, err
		}
		graph := packBranchGraph(edges, points)
		mst := branchMST(len(points), edges)
		if len(mst) != len(points)-1 {
			return BranchResult{}, fmt.Errorf("hdbscan: disconnected branch graph for cluster %d", cluster)
		}
		sort.SliceStable(mst, func(i, j int) bool { return lessMST(mst[i], mst[j]) })
		link := linkage(len(points), mst)
		tree := condense(link, cfg.MinClusterSize)
		selectCfg := Config{MinClusterSize: cfg.MinClusterSize, ClusterSelectionMethod: cfg.ClusterSelectionMethod,
			AllowSingleCluster: cfg.AllowSingleCluster, MaxClusterSize: cfg.MaxClusterSize,
			ClusterSelectionEpsilon: cfg.ClusterSelectionEpsilon, ClusterSelectionPersistence: cfg.ClusterSelectionPersistence}
		labels, probs, persistence := selectClusters(len(points), tree, stabilities(tree), selectCfg)
		out.ClusterPoints = append(out.ClusterPoints, append([]int(nil), points...))
		out.ApproximationGraphs = append(out.ApproximationGraphs, graph)
		out.LinkageTrees = append(out.LinkageTrees, link)
		out.CondensedTrees = append(out.CondensedTrees, tree)
		out.BranchPersistences = append(out.BranchPersistences, persistence)
		threshold := 2
		if cfg.LabelSidesAsBranches {
			threshold = 1
		}
		if len(persistence) <= threshold {
			for _, point := range points {
				out.Labels[point] = running
			}
			running++
		} else {
			hasNoise := 0
			for _, label := range labels {
				if label < 0 {
					hasNoise = 1
					break
				}
			}
			for local, point := range points {
				out.Labels[point] = labels[local] + running + hasNoise
				out.BranchLabels[point] = labels[local]
				out.BranchProbabilities[point] = probs[local]
				out.Probabilities[point] = (out.Probabilities[point] + probs[local]) / 2
			}
			running += len(persistence) + hasNoise
		}
	}
	return out, nil
}

func normalizeBranchConfig(c BranchConfig, fitted Config) (BranchConfig, error) {
	if c.Method == "" {
		c.Method = BranchFull
	}
	if c.Method != BranchFull && c.Method != BranchCore {
		return c, fmt.Errorf("%w: invalid branch detection method %q", ErrInvalidConfig, c.Method)
	}
	if c.MinClusterSize == 0 {
		c.MinClusterSize = fitted.MinClusterSize
	}
	if c.MinClusterSize < 2 {
		return c, fmt.Errorf("%w: branch min cluster size must be at least 2", ErrInvalidConfig)
	}
	if c.ClusterSelectionMethod == "" {
		c.ClusterSelectionMethod = fitted.ClusterSelectionMethod
	}
	if c.ClusterSelectionMethod == "" {
		c.ClusterSelectionMethod = EOM
	}
	if c.ClusterSelectionMethod != EOM && c.ClusterSelectionMethod != Leaf {
		return c, fmt.Errorf("%w: invalid branch selection method", ErrInvalidConfig)
	}
	if c.MaxClusterSize < 0 || c.ClusterSelectionEpsilon < 0 || math.IsNaN(c.ClusterSelectionEpsilon) || math.IsInf(c.ClusterSelectionEpsilon, 0) || c.ClusterSelectionPersistence < 0 || c.ClusterSelectionPersistence > 1 || math.IsNaN(c.ClusterSelectionPersistence) {
		return c, fmt.Errorf("%w: invalid branch selection threshold", ErrInvalidConfig)
	}
	if c.ClusterLabels != nil {
		if c.ClusterProbabilities != nil && len(c.ClusterProbabilities) != len(c.ClusterLabels) {
			return c, fmt.Errorf("%w: cluster probability length differs from labels", ErrInvalidConfig)
		}
		maxLabel := -1
		seen := map[int]bool{}
		for i, label := range c.ClusterLabels {
			if label < -1 {
				return c, fmt.Errorf("%w: invalid cluster label at %d", ErrInvalidConfig, i)
			}
			if label >= 0 {
				seen[label] = true
				maxLabel = max(maxLabel, label)
			}
		}
		for label := 0; label <= maxLabel; label++ {
			if !seen[label] {
				return c, fmt.Errorf("%w: cluster labels must be contiguous", ErrInvalidConfig)
			}
		}
		for i, p := range c.ClusterProbabilities {
			if math.IsNaN(p) || p < 0 || p > 1 {
				return c, fmt.Errorf("%w: invalid cluster probability at %d", ErrInvalidConfig, i)
			}
		}
	}
	return c, nil
}

func clusterCentrality(b *branchDetectionData, points []int, probability []float64) []float64 {
	center := make([]float64, b.raw.Cols)
	total := 0.0
	for _, p := range points {
		w := probability[p]
		total += w
		for d, v := range b.raw.Row(p) {
			center[d] += w * v
		}
	}
	if total == 0 {
		total = float64(len(points))
		for _, p := range points {
			for d, v := range b.raw.Row(p) {
				center[d] += v
			}
		}
	}
	for i := range center {
		center[i] /= total
	}
	out := make([]float64, len(points))
	for i, p := range points {
		d := b.metric.Distance(center, b.raw.Row(p))
		if d == 0 {
			out[i] = math.Inf(1)
		} else {
			out[i] = 1 / d
		}
	}
	return out
}

func buildBranchGraph(ctx context.Context, b *branchDetectionData, mst []MSTEdge, points []int, centrality []float64, method BranchDetectionMethod) ([]branchEdge, error) {
	local := make([]int, b.raw.Rows)
	for i := range local {
		local[i] = -1
	}
	for i, p := range points {
		local[p] = i
	}
	seen := make(map[uint64]branchEdge)
	add := func(a, c int, reach float64) {
		if a == c || a < 0 || c < 0 {
			return
		}
		if a > c {
			a, c = c, a
		}
		key := uint64(uint32(a))<<32 | uint64(uint32(c))
		e := branchEdge{a, c, max(centrality[a], centrality[c]), reach}
		if old, ok := seen[key]; !ok || reach < old.reachability {
			seen[key] = e
		}
	}
	maxDist := 0.0
	for _, e := range mst {
		a, c := local[e.From], local[e.To]
		if a >= 0 && c >= 0 {
			if method == BranchCore {
				add(a, c, e.Distance)
			}
			maxDist = max(maxDist, e.Distance)
		}
	}
	if method == BranchCore {
		for li, p := range points {
			for _, q := range b.neighbors[p*b.k : (p+1)*b.k] {
				lj := local[q]
				if lj < 0 {
					continue
				}
				d := b.metric.Distance(b.raw.Row(p), b.raw.Row(q))
				add(li, lj, max(d, b.core[p], b.core[q]))
			}
		}
	} else {
		for i, p := range points {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			for j := i + 1; j < len(points); j++ {
				q := points[j]
				d := b.metric.Distance(b.raw.Row(p), b.raw.Row(q))
				reach := max(d, b.core[p], b.core[q])
				if reach <= maxDist {
					add(i, j, reach)
				}
			}
		}
	}
	out := make([]branchEdge, 0, len(seen))
	for _, e := range seen {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].a != out[j].a {
			return out[i].a < out[j].a
		}
		return out[i].b < out[j].b
	})
	return out, nil
}

func packBranchGraph(edges []branchEdge, points []int) BranchGraph {
	g := BranchGraph{Packed: make([]uint64, len(edges)), Centrality: make([]float64, len(edges)), Reachability: make([]float64, len(edges))}
	for i, e := range edges {
		a, c := points[e.a], points[e.b]
		if a > c {
			a, c = c, a
		}
		g.Packed[i] = uint64(uint32(a))<<32 | uint64(uint32(c))
		g.Centrality[i] = e.centrality
		g.Reachability[i] = e.reachability
	}
	return g
}

func branchMST(n int, edges []branchEdge) []MSTEdge {
	sorted := append([]branchEdge(nil), edges...)
	sort.SliceStable(sorted, func(i, j int) bool {
		a, c := sorted[i], sorted[j]
		if a.centrality != c.centrality {
			return a.centrality < c.centrality
		}
		if a.reachability != c.reachability {
			return a.reachability < c.reachability
		}
		if a.a != c.a {
			return a.a < c.a
		}
		return a.b < c.b
	})
	u := newPlain(n)
	out := make([]MSTEdge, 0, n-1)
	for _, e := range sorted {
		if u.find(e.a) != u.find(e.b) {
			u.union(e.a, e.b)
			out = append(out, MSTEdge{e.a, e.b, e.centrality})
			if len(out) == n-1 {
				break
			}
		}
	}
	return out
}

func lessMST(a, b MSTEdge) bool {
	if a.Distance != b.Distance {
		return a.Distance < b.Distance
	}
	aa, ab := min(a.From, a.To), max(a.From, a.To)
	ba, bb := min(b.From, b.To), max(b.From, b.To)
	if aa != ba {
		return aa < ba
	}
	return ab < bb
}

// ApproximatePredict assigns new observations to both fitted clusters and
// detected branches, following the branch of the nearest connecting point.
func (b BranchResult) ApproximatePredict(ctx context.Context, points Dense64) (labels []int, probabilities []float64, clusterLabels []int, clusterProbabilities []float64, branchLabels []int, branchProbabilities []float64, err error) {
	p, e := b.source.predictionData()
	if e != nil {
		err = e
		return
	}
	if e = p.validate(points); e != nil {
		err = e
		return
	}
	n := points.Rows
	labels = make([]int, n)
	probabilities = make([]float64, n)
	clusterLabels = make([]int, n)
	clusterProbabilities = make([]float64, n)
	branchLabels = make([]int, n)
	branchProbabilities = make([]float64, n)
	s := newPredictScratch(p.raw.Rows)
	for i := 0; i < n; i++ {
		if e = ctx.Err(); e != nil {
			err = e
			return
		}
		neighbor, lambda := p.neighborLambda(points.Row(i), s)
		cl, cp := p.labelProbability(neighbor, lambda)
		clusterLabels[i], clusterProbabilities[i] = cl, cp
		branchProbabilities[i] = 1
		if cl < 0 {
			labels[i] = -1
			continue
		}
		count := 0
		if cl < len(b.BranchPersistences) {
			count = len(b.BranchPersistences[cl])
		}
		threshold := 2
		if b.labelSides {
			threshold = 1
		}
		if count <= threshold {
			labels[i], probabilities[i] = cl, cp
		} else {
			labels[i] = b.Labels[neighbor]
			branchLabels[i] = b.BranchLabels[neighbor]
			branchProbabilities[i] = b.BranchProbabilities[neighbor]
			probabilities[i] = (cp + branchProbabilities[i]) / 2
		}
	}
	return
}

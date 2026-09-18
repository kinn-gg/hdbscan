package hdbscan

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
)

// ClusterSelectionMethod controls how flat clusters are extracted.
type ClusterSelectionMethod string

const (
	EOM  ClusterSelectionMethod = "eom"
	Leaf ClusterSelectionMethod = "leaf"
)

// Config configures the Reference implementation. Zero values select the
// upstream defaults (MinClusterSize=5, MinSamples=MinClusterSize, Alpha=1).
type Config struct {
	MinClusterSize         int
	MinSamples             int
	Alpha                  float64
	Metric                 Metric
	ClusterSelectionMethod ClusterSelectionMethod
	AllowSingleCluster     bool
	MaxClusterSize         int
	// Workers bounds parallel work in Exact. Zero uses GOMAXPROCS.
	Workers int
}

// MSTEdge is an edge in the mutual-reachability minimum spanning tree.
type MSTEdge struct {
	From, To int
	Distance float64
}

// Linkage records one merge in the single-linkage hierarchy.
type Linkage struct {
	Left, Right int
	Distance    float64
	Size        int
}

// CondensedEdge records a parent-child relation in the condensed hierarchy.
type CondensedEdge struct {
	Parent, Child int
	Lambda        float64
	ChildSize     int
}

// Result owns all output slices produced by Reference.
type Result struct {
	Labels              []int
	Probabilities       []float64
	ClusterPersistence  []float64
	OutlierScores       []float64
	MinimumSpanningTree []MSTEdge
	SingleLinkageTree   []Linkage
	CondensedTree       []CondensedEdge
}

var ErrTooFewPoints = errors.New("hdbscan: at least two points are required")

// Reference runs the deliberately simple exact HDBSCAN* correctness oracle.
// It uses O(n²) time and O(n²) memory to materialize pairwise distances.
func Reference(ctx context.Context, x Dense64, cfg Config) (Result, error) {
	if err := x.Validate(); err != nil {
		return Result{}, err
	}
	if x.Rows < 2 {
		return Result{}, ErrTooFewPoints
	}
	metric := cfg.Metric
	if metric == nil {
		metric = Euclidean
	}
	n := x.Rows
	d := make([]float64, n*n)
	for i := 0; i < n; i++ {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		for j := 0; j < i; j++ {
			v := metric.Distance(x.Row(i), x.Row(j))
			if math.IsNaN(v) || v < 0 {
				return Result{}, fmt.Errorf("hdbscan: metric returned invalid distance")
			}
			d[i*n+j], d[j*n+i] = v, v
		}
	}
	return referenceDistances(ctx, d, n, cfg)
}

// ReferencePrecomputed runs Reference with a caller-supplied dense distance
// matrix. Positive infinity is allowed for missing edges.
func ReferencePrecomputed(ctx context.Context, p Precomputed, cfg Config) (Result, error) {
	if err := p.Validate(); err != nil {
		return Result{}, err
	}
	if p.Rows < 2 {
		return Result{}, ErrTooFewPoints
	}
	return referenceDistances(ctx, append([]float64(nil), p.Data...), p.Rows, cfg)
}

func normalizeConfig(c Config, n int) (Config, error) {
	if c.MinClusterSize == 0 {
		c.MinClusterSize = 5
	}
	if c.MinClusterSize < 2 {
		return c, fmt.Errorf("hdbscan: min cluster size must be at least 2")
	}
	if c.MinSamples == 0 {
		c.MinSamples = c.MinClusterSize
	}
	if c.MinSamples < 1 {
		return c, fmt.Errorf("hdbscan: min samples must be positive")
	}
	if c.Alpha == 0 {
		c.Alpha = 1
	}
	if c.Alpha < 0 || math.IsNaN(c.Alpha) || math.IsInf(c.Alpha, 0) {
		return c, fmt.Errorf("hdbscan: alpha must be finite and positive")
	}
	if c.ClusterSelectionMethod == "" {
		c.ClusterSelectionMethod = EOM
	}
	if c.ClusterSelectionMethod != EOM && c.ClusterSelectionMethod != Leaf {
		return c, fmt.Errorf("hdbscan: invalid cluster selection method %q", c.ClusterSelectionMethod)
	}
	return c, nil
}

func referenceDistances(ctx context.Context, d []float64, n int, cfg Config) (Result, error) {
	cfg, err := normalizeConfig(cfg, n)
	if err != nil {
		return Result{}, err
	}
	k := cfg.MinSamples
	if k > n-1 {
		k = n - 1
	}
	core := make([]float64, n)
	scratch := make([]float64, n)
	for i := 0; i < n; i++ {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		copy(scratch, d[i*n:(i+1)*n])
		sort.Float64s(scratch)
		core[i] = scratch[k]
	}
	mr := func(a, b int) float64 {
		v := d[a*n+b] / cfg.Alpha
		if core[a] > v {
			v = core[a]
		}
		if core[b] > v {
			v = core[b]
		}
		return v
	}
	mst := densePrim(ctx, n, mr)
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	for _, e := range mst {
		if math.IsInf(e.Distance, 1) {
			return Result{}, fmt.Errorf("hdbscan: distance graph is disconnected")
		}
	}
	sorted := append([]MSTEdge(nil), mst...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Distance < sorted[j].Distance })
	link := linkage(n, sorted)
	condensed := condense(link, cfg.MinClusterSize)
	stability := stabilities(condensed)
	labels, probs, persistence := selectClusters(n, condensed, stability, cfg)
	return Result{labels, probs, persistence, outliers(n, condensed), mst, link, condensed}, nil
}

func densePrim(ctx context.Context, n int, dist func(int, int) float64) []MSTEdge {
	in := make([]bool, n)
	best := make([]float64, n)
	src := make([]int, n)
	for i := range best {
		best[i] = math.Inf(1)
	}
	result := make([]MSTEdge, 0, n-1)
	cur := 0
	for step := 1; step < n; step++ {
		if ctx.Err() != nil {
			return result
		}
		in[cur] = true
		nd := math.Inf(1)
		next, from := 0, 0
		for j := 0; j < n; j++ {
			if in[j] {
				continue
			}
			v := dist(cur, j)
			if v < best[j] {
				best[j], src[j] = v, cur
			}
			if best[j] < nd {
				nd, next, from = best[j], j, src[j]
			}
		}
		result = append(result, MSTEdge{from, next, nd})
		cur = next
	}
	return result
}

type mergeUF struct {
	parent, size []int
	next         int
}

func newMergeUF(n int) *mergeUF {
	p := make([]int, 2*n-1)
	s := make([]int, 2*n-1)
	for i := range p {
		p[i] = -1
	}
	for i := 0; i < n; i++ {
		s[i] = 1
	}
	return &mergeUF{p, s, n}
}
func (u *mergeUF) find(x int) int {
	r := x
	for u.parent[r] >= 0 {
		r = u.parent[r]
	}
	for x != r && u.parent[x] >= 0 {
		q := u.parent[x]
		u.parent[x] = r
		x = q
	}
	return r
}
func (u *mergeUF) union(a, b int) {
	u.size[u.next] = u.size[a] + u.size[b]
	u.parent[a], u.parent[b] = u.next, u.next
	u.next++
}
func linkage(n int, edges []MSTEdge) []Linkage {
	u := newMergeUF(n)
	r := make([]Linkage, 0, n-1)
	for _, e := range edges {
		a, b := u.find(e.From), u.find(e.To)
		r = append(r, Linkage{a, b, e.Distance, u.size[a] + u.size[b]})
		u.union(a, b)
	}
	return r
}

func hierarchyBFS(h []Linkage, root, n int) []int {
	q := []int{root}
	out := []int{}
	for len(q) > 0 {
		x := q[0]
		q = q[1:]
		out = append(out, x)
		if x >= n {
			r := h[x-n]
			q = append(q, r.Left, r.Right)
		}
	}
	return out
}
func condense(h []Linkage, minSize int) []CondensedEdge {
	n := len(h) + 1
	root := 2*n - 2
	rel := make([]int, 2*n-1)
	rel[root] = n
	next := n + 1
	ignore := make([]bool, 2*n-1)
	out := make([]CondensedEdge, 0, 2*n)
	for _, node := range hierarchyBFS(h, root, n) {
		if node < n || ignore[node] {
			continue
		}
		m := h[node-n]
		lambda := math.Inf(1)
		if m.Distance > 0 {
			lambda = 1 / m.Distance
		}
		count := func(x int) int {
			if x < n {
				return 1
			}
			return h[x-n].Size
		}
		lc, rc := count(m.Left), count(m.Right)
		addPoints := func(x int) {
			for _, s := range hierarchyBFS(h, x, n) {
				if s < n {
					out = append(out, CondensedEdge{rel[node], s, lambda, 1})
				}
				ignore[s] = true
			}
		}
		switch {
		case lc >= minSize && rc >= minSize:
			rel[m.Left] = next
			next++
			rel[m.Right] = next
			next++
			out = append(out, CondensedEdge{rel[node], rel[m.Left], lambda, lc}, CondensedEdge{rel[node], rel[m.Right], lambda, rc})
		case lc < minSize && rc < minSize:
			addPoints(m.Left)
			addPoints(m.Right)
		case lc < minSize:
			rel[m.Right] = rel[node]
			addPoints(m.Left)
		default:
			rel[m.Left] = rel[node]
			addPoints(m.Right)
		}
	}
	return out
}

func stabilities(t []CondensedEdge) map[int]float64 {
	birth := map[int]float64{}
	for _, e := range t {
		v, ok := birth[e.Child]
		if !ok || e.Lambda < v {
			birth[e.Child] = e.Lambda
		}
	}
	if len(t) > 0 {
		birth[t[0].Parent] = 0
	}
	s := map[int]float64{}
	for _, e := range t {
		s[e.Parent] += (e.Lambda - birth[e.Parent]) * float64(e.ChildSize)
	}
	return s
}
func maxLambdas(t []CondensedEdge, size int) []float64 {
	d := make([]float64, size)
	for _, e := range t {
		if e.Lambda > d[e.Parent] {
			d[e.Parent] = e.Lambda
		}
	}
	return d
}

type plainUF struct{ p, r []int }

func newPlain(n int) *plainUF { p := make([]int, n); return &plainUF{p, make([]int, n)} }
func (u *plainUF) find(x int) int {
	if u.p[x] == 0 {
		u.p[x] = x + 1
	}
	p := u.p[x] - 1
	if p != x {
		p = u.find(p)
		u.p[x] = p + 1
	}
	return p
}
func (u *plainUF) union(a, b int) {
	a, b = u.find(a), u.find(b)
	if a == b {
		return
	}
	if u.r[a] < u.r[b] {
		a, b = b, a
	}
	u.p[b] = a + 1
	if u.r[a] == u.r[b] {
		u.r[a]++
	}
}
func selectClusters(n int, t []CondensedEdge, s map[int]float64, c Config) ([]int, []float64, []float64) {
	labels := make([]int, n)
	for i := range labels {
		labels[i] = -1
	}
	probs := make([]float64, n)
	if len(t) == 0 {
		return labels, probs, nil
	}
	children := map[int][]int{}
	sizes := map[int]int{}
	maxID := n
	for _, e := range t {
		if e.ChildSize > 1 {
			children[e.Parent] = append(children[e.Parent], e.Child)
			sizes[e.Child] = e.ChildSize
		}
		if e.Parent > maxID {
			maxID = e.Parent
		}
		if e.Child > maxID {
			maxID = e.Child
		}
	}
	root := t[0].Parent
	selected := map[int]bool{}
	if c.ClusterSelectionMethod == Leaf {
		for id := range s {
			if id != root && len(children[id]) == 0 {
				selected[id] = true
			}
		}
	} else {
		ids := make([]int, 0, len(s))
		for id := range s {
			ids = append(ids, id)
		}
		sort.Sort(sort.Reverse(sort.IntSlice(ids)))
		for _, id := range ids {
			if id == root && !c.AllowSingleCluster {
				continue
			}
			sub := 0.0
			for _, ch := range children[id] {
				sub += s[ch]
			}
			if sub > s[id] || (c.MaxClusterSize > 0 && sizes[id] > c.MaxClusterSize) {
				s[id] = sub
			} else {
				selected[id] = true
				var clear func(int)
				clear = func(x int) {
					for _, ch := range children[x] {
						delete(selected, ch)
						clear(ch)
					}
				}
				clear(id)
			}
		}
	}
	clusters := make([]int, 0)
	for id := range selected {
		clusters = append(clusters, id)
	}
	sort.Ints(clusters)
	cmap := map[int]int{}
	for i, id := range clusters {
		cmap[id] = i
	}
	u := newPlain(maxID + 1)
	childLambda := make([]float64, maxID+1)
	parentMaxLambda := make([]float64, maxID+1)
	for _, e := range t {
		childLambda[e.Child] = e.Lambda
		if e.Lambda > parentMaxLambda[e.Parent] {
			parentMaxLambda[e.Parent] = e.Lambda
		}
		if !selected[e.Child] {
			u.union(e.Parent, e.Child)
		}
	}
	for i := 0; i < n; i++ {
		cl := u.find(i)
		if v, ok := cmap[cl]; ok {
			if cl != root || len(clusters) != 1 || !c.AllowSingleCluster || childLambda[i] >= parentMaxLambda[root] {
				labels[i] = v
			}
		}
	}
	deaths := maxLambdas(t, maxID+1)
	for i := len(t) - 1; i >= 0; i-- {
		e := t[i]
		if deaths[e.Child] > deaths[e.Parent] {
			deaths[e.Parent] = deaths[e.Child]
		}
	}
	counts := make([]int, len(clusters))
	for _, v := range labels {
		if v >= 0 {
			counts[v]++
		}
	}
	for _, e := range t {
		if e.Child >= n || labels[e.Child] < 0 {
			continue
		}
		cl := clusters[labels[e.Child]]
		m := deaths[cl]
		if m == 0 || math.IsInf(e.Lambda, 0) {
			probs[e.Child] = 1
		} else {
			probs[e.Child] = math.Min(e.Lambda, m) / m
		}
	}
	persist := make([]float64, len(clusters))
	maxLam := 0.0
	for _, e := range t {
		if e.Lambda > maxLam {
			maxLam = e.Lambda
		}
	}
	for i, id := range clusters {
		if math.IsInf(maxLam, 0) || maxLam == 0 || counts[i] == 0 {
			persist[i] = 1
		} else {
			persist[i] = s[id] / (float64(counts[i]) * maxLam)
		}
	}
	return labels, probs, persist
}
func outliers(n int, t []CondensedEdge) []float64 {
	r := make([]float64, n)
	if len(t) == 0 {
		return r
	}
	max := 0
	for _, e := range t {
		if e.Parent > max {
			max = e.Parent
		}
		if e.Child > max {
			max = e.Child
		}
	}
	d := maxLambdas(t, max+1)
	for i := len(t) - 1; i >= 0; i-- {
		e := t[i]
		if d[e.Child] > d[e.Parent] {
			d[e.Parent] = d[e.Child]
		}
	}
	for _, e := range t {
		if e.Child < n {
			m := d[e.Parent]
			if m != 0 && !math.IsInf(e.Lambda, 0) {
				r[e.Child] = (m - e.Lambda) / m
			}
		}
	}
	return r
}

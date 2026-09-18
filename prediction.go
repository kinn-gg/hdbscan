package hdbscan

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
)

var (
	ErrPredictionData        = errors.New("hdbscan: prediction data was not retained")
	ErrPredictionUnsupported = errors.New("hdbscan: prediction is unsupported for precomputed distances")
)

type predictionData struct {
	raw                Dense64
	core               []float64
	metric             Metric
	minSamples         int
	clusters           []int
	clusterMap         map[int]int
	maxLambda, leafMax map[int]float64
	parent             map[int]CondensedEdge
	pointEdge          []CondensedEdge
	exemplars          [][]int
	tree               []CondensedEdge
}

func makePredictionData(x Dense64, tree []CondensedEdge, cfg Config) *predictionData {
	if !cfg.PredictionData || x.Rows == 0 {
		return nil
	}
	p := &predictionData{
		raw:        Dense64{Data: append([]float64(nil), x.Data...), Rows: x.Rows, Cols: x.Cols},
		minSamples: min(cfg.MinSamples, x.Rows-1),
		metric:     cfg.Metric, tree: append([]CondensedEdge(nil), tree...),
		maxLambda: map[int]float64{}, leafMax: map[int]float64{}, parent: map[int]CondensedEdge{},
		pointEdge: make([]CondensedEdge, x.Rows),
	}
	if p.metric == nil {
		p.metric = Euclidean
	}
	p.core = predictionCoreDistances(p.raw, p.metric, cfg.MinSamples)
	p.clusters = predictionClusters(tree, cfg)
	p.clusterMap = make(map[int]int, len(p.clusters))
	children := map[int][]int{}
	for _, e := range tree {
		if e.Lambda > p.leafMax[e.Parent] {
			p.leafMax[e.Parent] = e.Lambda
		}
		if e.ChildSize > 1 {
			p.parent[e.Child] = e
			children[e.Parent] = append(children[e.Parent], e.Child)
		} else if e.Child < x.Rows {
			p.pointEdge[e.Child] = e
		}
	}
	for label, cluster := range p.clusters {
		p.clusterMap[cluster] = label
		p.maxLambda[cluster] = p.leafMax[cluster]
		stack := []int{cluster}
		for len(stack) > 0 {
			id := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			p.clusterMap[id], p.maxLambda[id] = label, p.leafMax[cluster]
			stack = append(stack, children[id]...)
		}
		leaves := []int{}
		stack = []int{cluster}
		for len(stack) > 0 {
			id := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if len(children[id]) == 0 {
				leaves = append(leaves, id)
			} else {
				stack = append(stack, children[id]...)
			}
		}
		var exemplars []int
		for _, leaf := range leaves {
			m := p.leafMax[leaf]
			for _, e := range tree {
				if e.Parent == leaf && e.ChildSize == 1 && e.Lambda == m {
					exemplars = append(exemplars, e.Child)
				}
			}
		}
		p.exemplars = append(p.exemplars, exemplars)
	}
	return p
}

func predictionCoreDistances(x Dense64, metric Metric, minSamples int) []float64 {
	core, scratch := make([]float64, x.Rows), make([]float64, x.Rows)
	k := min(max(minSamples, 1), x.Rows) - 1
	for i := 0; i < x.Rows; i++ {
		for j := 0; j < x.Rows; j++ {
			scratch[j] = metric.Distance(x.Row(i), x.Row(j))
		}
		sort.Float64s(scratch)
		core[i] = scratch[k]
	}
	return core
}

// predictionClusters returns the selected hierarchy ids in label order.
func predictionClusters(t []CondensedEdge, cfg Config) []int {
	if len(t) == 0 {
		return nil
	}
	s := stabilities(t)
	children, sizes := map[int][]int{}, map[int]int{}
	root := t[0].Parent
	for _, e := range t {
		if e.ChildSize > 1 {
			children[e.Parent] = append(children[e.Parent], e.Child)
			sizes[e.Child] = e.ChildSize
		}
	}
	selected := map[int]bool{}
	if cfg.ClusterSelectionMethod == Leaf {
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
			if id == root && !cfg.AllowSingleCluster {
				continue
			}
			sub := 0.0
			for _, ch := range children[id] {
				sub += s[ch]
			}
			if sub > s[id] || (cfg.MaxClusterSize > 0 && sizes[id] > cfg.MaxClusterSize) {
				s[id] = sub
				continue
			}
			selected[id] = true
			stack := append([]int(nil), children[id]...)
			for len(stack) > 0 {
				ch := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				delete(selected, ch)
				stack = append(stack, children[ch]...)
			}
		}
	}
	if cfg.ClusterSelectionEpsilon > 0 {
		birth := map[int]float64{}
		up := map[int]int{}
		for _, e := range t {
			if e.ChildSize > 1 {
				birth[e.Child], up[e.Child] = e.Lambda, e.Parent
			}
		}
		merged := map[int]bool{}
		threshold := 1 / cfg.ClusterSelectionEpsilon
		for id := range selected {
			target := id
			for birth[target] > threshold {
				next, ok := up[target]
				if !ok || (next == root && !cfg.AllowSingleCluster) {
					break
				}
				target = next
			}
			merged[target] = true
		}
		selected = merged
	}
	clusters := make([]int, 0, len(selected))
	for id := range selected {
		clusters = append(clusters, id)
	}
	sort.Ints(clusters)
	return clusters
}

func (r Result) predictionData() (*predictionData, error) {
	if r.prediction == nil {
		return nil, ErrPredictionData
	}
	return r.prediction, nil
}

// ClusterCount returns the number of columns produced by membership methods.
func (r Result) ClusterCount() int {
	if r.prediction == nil {
		return len(r.ClusterPersistence)
	}
	return len(r.prediction.clusters)
}

// ApproximatePredict assigns new observations without changing the fitted hierarchy.
func (r Result) ApproximatePredict(ctx context.Context, points Dense64) ([]int, []float64, error) {
	labels, strengths := make([]int, points.Rows), make([]float64, points.Rows)
	if err := r.PredictInto(ctx, points, labels, strengths); err != nil {
		return nil, nil, err
	}
	return labels, strengths, nil
}

// PredictInto is the caller-buffer form of ApproximatePredict.
func (r Result) PredictInto(ctx context.Context, points Dense64, labels []int, strengths []float64) error {
	p, err := r.predictionData()
	if err != nil {
		return err
	}
	if err = p.validate(points); err != nil {
		return err
	}
	if len(labels) < points.Rows || len(strengths) < points.Rows {
		return fmt.Errorf("hdbscan: prediction output buffer too small")
	}
	if len(p.clusters) == 0 {
		for i := 0; i < points.Rows; i++ {
			labels[i] = -1
			strengths[i] = 0
		}
		return nil
	}
	s := newPredictScratch(p.raw.Rows)
	for i := 0; i < points.Rows; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		neighbor, lambda := p.neighborLambda(points.Row(i), s)
		labels[i], strengths[i] = p.labelProbability(neighbor, lambda)
	}
	return nil
}

// ApproximatePredictScores estimates GLOSH outlier scores for new observations.
func (r Result) ApproximatePredictScores(ctx context.Context, points Dense64) ([]float64, error) {
	out := make([]float64, points.Rows)
	if err := r.PredictScoresInto(ctx, points, out); err != nil {
		return nil, err
	}
	return out, nil
}

// PredictScoresInto writes approximate outlier scores into a caller buffer.
func (r Result) PredictScoresInto(ctx context.Context, points Dense64, dst []float64) error {
	p, err := r.predictionData()
	if err != nil {
		return err
	}
	if err = p.validate(points); err != nil {
		return err
	}
	if len(dst) < points.Rows {
		return fmt.Errorf("hdbscan: prediction output buffer too small")
	}
	if len(p.clusters) == 0 {
		for i := 0; i < points.Rows; i++ {
			dst[i] = 1
		}
		return nil
	}
	maxima := make(map[int]float64, len(p.leafMax))
	for k, v := range p.leafMax {
		maxima[k] = v
	}
	for i := len(p.tree) - 1; i >= 0; i-- {
		e := p.tree[i]
		if maxima[e.Child] > maxima[e.Parent] {
			maxima[e.Parent] = maxima[e.Child]
		}
	}
	s := newPredictScratch(p.raw.Rows)
	for i := 0; i < points.Rows; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		neighbor, lambda := p.neighborLambda(points.Row(i), s)
		edge := p.pointEdge[neighbor]
		if s.minDistance == 0 {
			lambda = edge.Lambda
		}
		m := maxima[edge.Parent]
		if m > 0 {
			dst[i] = (m - lambda) / m
		} else {
			dst[i] = 0
		}
	}
	return nil
}

// MembershipVector computes one new observation's soft membership vector.
func (r Result) MembershipVector(ctx context.Context, point []float64) ([]float64, error) {
	p, err := r.predictionData()
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(p.clusters))
	err = r.MembershipVectorsInto(ctx, Dense64{Data: point, Rows: 1, Cols: len(point)}, out)
	return out, err
}

// MembershipVectors computes row-major soft membership vectors for new rows.
func (r Result) MembershipVectors(ctx context.Context, points Dense64) (Dense64, error) {
	p, err := r.predictionData()
	if err != nil {
		return Dense64{}, err
	}
	out := Dense64{Data: make([]float64, points.Rows*len(p.clusters)), Rows: points.Rows, Cols: len(p.clusters)}
	if err := r.MembershipVectorsInto(ctx, points, out.Data); err != nil {
		return Dense64{}, err
	}
	return out, nil
}

// MembershipVectorsInto writes row-major soft memberships to dst. Scratch memory
// is O(training rows + clusters), independent of the query batch size.
func (r Result) MembershipVectorsInto(ctx context.Context, points Dense64, dst []float64) error {
	p, err := r.predictionData()
	if err != nil {
		return err
	}
	if err = p.validate(points); err != nil {
		return err
	}
	cols := len(p.clusters)
	if len(dst) < points.Rows*cols {
		return fmt.Errorf("hdbscan: membership output buffer too small")
	}
	if cols == 0 {
		return nil
	}
	s := newPredictScratch(p.raw.Rows)
	work := make([]float64, cols*3)
	for i := 0; i < points.Rows; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		neighbor, lambda := p.neighborLambda(points.Row(i), s)
		edge := p.pointEdge[neighbor]
		if edge.Lambda <= lambda {
			lambda = edge.Lambda
		}
		p.membership(points.Row(i), neighbor, lambda, dst[i*cols:(i+1)*cols], work)
	}
	return nil
}

// AllPointsMembershipVectorsInto writes soft memberships for the fitted rows.
func (r Result) AllPointsMembershipVectorsInto(ctx context.Context, dst []float64) error {
	p, err := r.predictionData()
	if err != nil {
		return err
	}
	cols := len(p.clusters)
	if len(dst) < p.raw.Rows*cols {
		return fmt.Errorf("hdbscan: membership output buffer too small")
	}
	if cols == 0 {
		return nil
	}
	work := make([]float64, cols*3)
	for i := 0; i < p.raw.Rows; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		e := p.pointEdge[i]
		p.allPointMembership(p.raw.Row(i), e.Parent, e.Lambda, dst[i*cols:(i+1)*cols], work)
	}
	return nil
}

func (p *predictionData) allPointMembership(point []float64, pointCluster int, lambda float64, out, work []float64) {
	k := len(p.clusters)
	distv, outlier, heights := work[:k], work[k:2*k], work[2*k:3*k]
	dsum := 0.0
	for c, ex := range p.exemplars {
		d := math.Inf(1)
		for _, idx := range ex {
			d = min(d, p.metric.Distance(point, p.raw.Row(idx)))
		}
		if d == 0 {
			distv[c] = math.MaxFloat64 / float64(k)
		} else {
			distv[c] = 1 / d
		}
		dsum += distv[c]
	}
	for i := range distv {
		distv[i] /= dsum
	}
	p.mergeHeights(pointCluster, lambda, heights)
	osum := 0.0
	for i, h := range heights {
		if h <= 0 {
			outlier[i] = 1
		} else {
			outlier[i] = math.Exp(math.Exp(-(p.leafMax[pointCluster] + 1e-8) / h))
		}
		osum += outlier[i]
	}
	for i := range outlier {
		outlier[i] /= osum
	}
	best := 0
	for i := 1; i < k; i++ {
		if heights[i] > heights[best] {
			best = i
		}
	}
	prob := heights[best] / max(p.leafMax[p.clusters[best]], lambda, 1e-8)
	sum := 0.0
	for i := 0; i < k; i++ {
		out[i] = distv[i] * outlier[i]
		sum += out[i]
	}
	if sum > 0 {
		for i := range out {
			out[i] = out[i] / sum * prob
		}
	}
}

// AllPointsMembershipVectors returns row-major memberships for fitted rows.
func (r Result) AllPointsMembershipVectors(ctx context.Context) (Dense64, error) {
	p, err := r.predictionData()
	if err != nil {
		return Dense64{}, err
	}
	out := Dense64{Data: make([]float64, p.raw.Rows*len(p.clusters)), Rows: p.raw.Rows, Cols: len(p.clusters)}
	err = r.AllPointsMembershipVectorsInto(ctx, out.Data)
	return out, err
}

func (p *predictionData) validate(x Dense64) error {
	if err := x.Validate(); err != nil {
		return err
	}
	if x.Cols != p.raw.Cols {
		return fmt.Errorf("%w: prediction columns %d, want %d", ErrInvalidShape, x.Cols, p.raw.Cols)
	}
	return nil
}

type predictScratch struct {
	distances   []float64
	indices     []int
	minDistance float64
}

func newPredictScratch(n int) *predictScratch {
	return &predictScratch{make([]float64, n), make([]int, n), 0}
}

func (p *predictionData) neighborLambda(point []float64, s *predictScratch) (int, float64) {
	n := p.raw.Rows
	limit := min(2*p.minSamples, n)
	if limit < 1 {
		limit = 1
	}
	count := 0
	for i := 0; i < n; i++ {
		d := p.metric.Distance(point, p.raw.Row(i))
		s.distances[i] = d
		if count == limit {
			last := s.indices[limit-1]
			if d > s.distances[last] || (d == s.distances[last] && i > last) {
				continue
			}
		}
		pos := count
		if pos > limit-1 {
			pos = limit - 1
		}
		for pos > 0 {
			prev := s.indices[pos-1]
			if s.distances[prev] < d || (s.distances[prev] == d && prev < i) {
				break
			}
			pos--
		}
		if count < limit {
			copy(s.indices[pos+1:count+1], s.indices[pos:count])
			s.indices[pos] = i
			count++
		} else if pos < limit {
			copy(s.indices[pos+1:limit], s.indices[pos:limit-1])
			s.indices[pos] = i
		}
	}
	s.minDistance = s.distances[s.indices[0]]
	pos := min(p.minSamples, n-1)
	pointCore := s.distances[s.indices[pos]]
	best := math.Inf(1)
	neighbor := -1
	for j := 0; j < limit; j++ {
		idx := s.indices[j]
		mr := max(p.core[idx], pointCore, s.distances[idx])
		if mr < best {
			best, neighbor = mr, idx
		}
	}
	if best > 0 {
		return neighbor, 1 / best
	}
	return neighbor, math.MaxFloat64
}

func (p *predictionData) labelProbability(neighbor int, lambda float64) (int, float64) {
	e := p.pointEdge[neighbor]
	cluster := e.Parent
	if e.Lambda > lambda {
		root := p.tree[0].Parent
		for cluster > root {
			up, ok := p.parent[cluster]
			if !ok || up.Lambda < lambda {
				break
			}
			cluster = up.Parent
		}
	}
	label, ok := p.clusterMap[cluster]
	if !ok {
		return -1, 0
	}
	m := p.maxLambda[cluster]
	if m <= 0 {
		return label, 1
	}
	return label, min(m, lambda) / m
}

func (p *predictionData) membership(point []float64, neighbor int, lambda float64, out, work []float64) {
	k := len(p.clusters)
	distv, outlier, heights := work[:k], work[k:2*k], work[2*k:3*k]
	dsum := 0.0
	for c, ex := range p.exemplars {
		d := math.Inf(1)
		for _, idx := range ex {
			d = min(d, p.metric.Distance(point, p.raw.Row(idx)))
		}
		if d == 0 {
			distv[c] = math.MaxFloat64 / float64(k)
		} else {
			distv[c] = 1 / d
		}
		dsum += distv[c]
	}
	for i := range distv {
		distv[i] /= dsum
	}
	pointCluster := p.pointEdge[neighbor].Parent
	p.mergeHeights(pointCluster, lambda, heights)
	m := p.leafMax[pointCluster]
	mx := -math.MaxFloat64
	for i, h := range heights {
		den := m - h
		if den <= 0 {
			den = 1e-8
		}
		outlier[i] = m / den
		if outlier[i] > mx {
			mx = outlier[i]
		}
	}
	osum := 0.0
	for i := range outlier {
		outlier[i] = math.Exp(outlier[i] - mx)
		osum += outlier[i]
	}
	for i := range outlier {
		outlier[i] /= osum
	}
	best := 0
	for i := 1; i < k; i++ {
		if heights[i] > heights[best] {
			best = i
		}
	}
	prob := heights[best] / max(lambda, p.leafMax[p.clusters[best]], 1e-8)
	sum := 0.0
	for i := 0; i < k; i++ {
		out[i] = math.Sqrt(distv[i]) * outlier[i] * outlier[i]
		sum += out[i]
	}
	if sum > 0 {
		for i := range out {
			out[i] = out[i] / sum * prob
		}
	}
}

func (p *predictionData) mergeHeights(pointCluster int, lambda float64, out []float64) {
	for i, right0 := range p.clusters {
		left, right := pointCluster, right0
		tookL, tookR := false, false
		last := 0
		for left != right {
			if left > right {
				tookL = true
				last = left
				left = p.parent[left].Parent
			} else {
				tookR = true
				last = right
				right = p.parent[right].Parent
			}
		}
		if tookL && tookR {
			out[i] = p.parent[last].Lambda
		} else {
			out[i] = lambda
		}
	}
}

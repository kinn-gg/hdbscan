package hdbscan

import (
	"context"
	"fmt"
	"math"
	"sort"
)

// RobustSingleLinkageConfig configures robust single linkage. Cut is the
// linkage distance at which the hierarchy is flattened. K selects the core
// distance neighbor and defaults to 5; Alpha defaults to sqrt(2).
type RobustSingleLinkageConfig struct {
	Cut     float64
	K       int
	Alpha   float64
	Gamma   int
	Metric  Metric
	Workers int
}

// RobustSingleLinkageResult owns the flat labels and complete hierarchy.
type RobustSingleLinkageResult struct {
	Labels              []int
	SingleLinkageTree   []Linkage
	MinimumSpanningTree []MSTEdge
}

// RobustSingleLinkage builds the hierarchy with the shared bounded-memory
// neighbor and MST pipeline and cuts it at config.Cut. Components smaller than
// Gamma (default 5) are labelled noise.
func RobustSingleLinkage(ctx context.Context, x Dense64, config RobustSingleLinkageConfig) (RobustSingleLinkageResult, error) {
	if err := x.Validate(); err != nil {
		return RobustSingleLinkageResult{}, err
	}
	if x.Rows < 2 {
		return RobustSingleLinkageResult{}, ErrTooFewPoints
	}
	if config.K == 0 {
		config.K = 5
	}
	if config.Gamma == 0 {
		config.Gamma = 5
	}
	if config.Alpha == 0 {
		config.Alpha = math.Sqrt2
	}
	if config.K < 1 || config.Gamma < 1 || config.Cut < 0 || math.IsNaN(config.Cut) || math.IsInf(config.Cut, 0) || config.Alpha < 1 || math.IsNaN(config.Alpha) || math.IsInf(config.Alpha, 0) {
		return RobustSingleLinkageResult{}, fmt.Errorf("%w: invalid robust single linkage configuration", ErrInvalidConfig)
	}
	metric := config.Metric
	if metric == nil {
		metric = Euclidean
	}
	workers, err := normalizeWorkers(config.Workers, x.Rows)
	if err != nil {
		return RobustSingleLinkageResult{}, err
	}
	core, err := blockedCoreDistances(ctx, x, metric, config.K, workers)
	if err != nil {
		return RobustSingleLinkageResult{}, err
	}
	mst, err := streamedPrim(ctx, x, metric, core, config.Alpha)
	if err != nil {
		return RobustSingleLinkageResult{}, err
	}
	sorted := append([]MSTEdge(nil), mst...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Distance < sorted[j].Distance })
	tree := linkage(x.Rows, sorted)
	labels := cutLinkage(tree, config.Cut, config.Gamma)
	return RobustSingleLinkageResult{labels, tree, mst}, nil
}

// CutSingleLinkage extracts connected groups at a linkage distance without
// rebuilding neighbors or the MST.
func CutSingleLinkage(tree []Linkage, cut float64, minClusterSize int) ([]int, error) {
	if minClusterSize < 1 || cut < 0 || math.IsNaN(cut) || math.IsInf(cut, 0) {
		return nil, fmt.Errorf("%w: invalid hierarchy cut", ErrInvalidConfig)
	}
	return cutLinkage(tree, cut, minClusterSize), nil
}

func cutLinkage(tree []Linkage, cut float64, minSize int) []int {
	n := len(tree) + 1
	uf := newMSTUF(2*n - 1)
	for i, row := range tree {
		node := n + i
		if row.Distance < cut {
			uf.union(row.Left, node)
			uf.union(row.Right, node)
		}
	}
	sizes := make(map[int]int, n)
	for i := 0; i < n; i++ {
		sizes[uf.find(i)]++
	}
	ids := map[int]int{}
	next := 0
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		root := uf.find(i)
		if sizes[root] < minSize {
			labels[i] = -1
			continue
		}
		id, ok := ids[root]
		if !ok {
			id = next
			next++
			ids[root] = id
		}
		labels[i] = id
	}
	return labels
}

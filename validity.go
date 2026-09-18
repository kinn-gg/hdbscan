package hdbscan

import (
	"context"
	"fmt"
	"math"
)

// ValidityIndex computes the density-based clustering validation (DBCV) score
// and each cluster's contribution. Noise labels (<0) are excluded. Work is
// streamed over point pairs, using O(n+c^2) memory rather than an n*n matrix.
func ValidityIndex(ctx context.Context, x Dense64, labels []int, metric Metric) (float64, []float64, error) {
	if err := x.Validate(); err != nil {
		return 0, nil, err
	}
	if len(labels) != x.Rows {
		return 0, nil, fmt.Errorf("hdbscan: labels length %d, want %d", len(labels), x.Rows)
	}
	if metric == nil {
		metric = Euclidean
	}
	dims := x.Cols
	if dims < 1 {
		return 0, nil, fmt.Errorf("hdbscan: validity requires positive dimension")
	}
	maxLabel := -1
	counts := map[int]int{}
	for _, label := range labels {
		if label >= 0 {
			counts[label]++
			if label > maxLabel {
				maxLabel = label
			}
		}
	}
	if maxLabel < 0 {
		return 0, nil, nil
	}
	for label, count := range counts {
		if count < 2 {
			return 0, nil, fmt.Errorf("hdbscan: cluster %d has fewer than two points", label)
		}
	}
	core := make([]float64, x.Rows)
	for i, label := range labels {
		if label < 0 {
			continue
		}
		var sum float64
		for j, other := range labels {
			if i == j || other != label {
				continue
			}
			if err := ctx.Err(); err != nil {
				return 0, nil, err
			}
			d := metric.Distance(x.Row(i), x.Row(j))
			if d == 0 {
				sum = math.Inf(1)
				break
			}
			sum += math.Pow(1/d, float64(dims))
		}
		core[i] = math.Pow(sum/float64(counts[label]-1), -1/float64(dims))
	}
	c := maxLabel + 1
	sparseness := make([]float64, c)
	internal := make([][]int, c)
	separation := make([]float64, c*c)
	for i := range separation {
		separation[i] = math.Inf(1)
	}
	// Per-cluster Prim over implicit all-pairs mutual reachability avoids both
	// the raw distance matrix and per-cluster quadratic matrices.
	for label := 0; label < c; label++ {
		if counts[label] == 0 {
			continue
		}
		indices := make([]int, 0, counts[label])
		for i, v := range labels {
			if v == label {
				indices = append(indices, i)
			}
		}
		in := make([]bool, len(indices))
		best := make([]float64, len(indices))
		parent := make([]int, len(indices))
		source := make([]int, len(indices))
		degree := make([]int, len(indices))
		order := make([]int, 0, len(indices)-1)
		for i := range best {
			best[i] = math.Inf(1)
		}
		cur := 0
		for step := 1; step < len(indices); step++ {
			in[cur] = true
			for j := range indices {
				if in[j] {
					continue
				}
				d := metric.Distance(x.Row(indices[cur]), x.Row(indices[j]))
				d = math.Max(d, math.Max(core[indices[cur]], core[indices[j]]))
				if d < best[j] {
					best[j] = d
					source[j] = cur
				}
			}
			next := -1
			for j := range indices {
				if !in[j] && (next < 0 || best[j] < best[next]) {
					next = j
				}
			}
			parent[next] = source[next]
			order = append(order, next)
			cur = next
		}
		// Match upstream's deterministic source reconstruction for nearly tied
		// MST edges; topology affects which vertices are considered internal.
		seen := make([]bool, len(indices))
		for edge, target := range order {
			if edge > 0 {
				for candidate := 0; candidate < len(indices); candidate++ {
					if !seen[candidate] || candidate == target {
						continue
					}
					d := metric.Distance(x.Row(indices[candidate]), x.Row(indices[target]))
					d = math.Max(d, math.Max(core[indices[candidate]], core[indices[target]]))
					if math.Abs(d-best[target]) <= 1e-8+1e-5*math.Abs(best[target]) {
						parent[target] = candidate
						break
					}
				}
			}
			seen[parent[target]], seen[target] = true, true
			degree[target]++
			degree[parent[target]]++
		}
		for i, d := range degree {
			if d > 1 {
				internal[label] = append(internal[label], indices[i])
			}
		}
		allVertices := false
		if len(internal[label]) == 0 {
			internal[label] = append(internal[label], indices[0])
			allVertices = true
		}
		for _, target := range order {
			if (degree[target] > 1 && degree[parent[target]] > 1) || allVertices {
				if best[target] > sparseness[label] {
					sparseness[label] = best[target]
				}
			}
		}
	}
	for a := 0; a < c; a++ {
		for b := 0; b < a; b++ {
			if counts[a] == 0 || counts[b] == 0 {
				continue
			}
			for _, i := range internal[a] {
				for _, j := range internal[b] {
					d := metric.Distance(x.Row(i), x.Row(j))
					d = math.Max(d, math.Max(core[i], core[j]))
					if d < separation[a*c+b] {
						separation[a*c+b], separation[b*c+a] = d, d
					}
				}
			}
		}
	}
	per := make([]float64, c)
	var score float64
	for label, count := range counts {
		sep := math.Inf(1)
		for other := 0; other < c; other++ {
			if other != label && counts[other] > 0 && separation[label*c+other] < sep {
				sep = separation[label*c+other]
			}
		}
		if math.IsInf(sep, 1) {
			per[label] = 0
		} else {
			per[label] = (sep - sparseness[label]) / math.Max(sep, sparseness[label])
		}
		score += float64(count) / float64(x.Rows) * per[label]
	}
	return score, per, nil
}

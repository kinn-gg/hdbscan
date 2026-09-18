package hdbscan

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
)

// SparsePrecomputed is a symmetric CSR distance graph. Missing off-diagonal
// entries are absent edges, never zero-distance edges. Diagonal zeroes may be
// omitted. Input slices are not copied during validation.
type SparsePrecomputed struct {
	Data       []float64
	Indices    []int
	IndPtr     []int
	Rows, Cols int
}

var (
	ErrInvalidSparse         = errors.New("hdbscan: invalid sparse distance graph")
	ErrDisconnected          = errors.New("hdbscan: distance graph is disconnected")
	ErrInsufficientNeighbors = errors.New("hdbscan: sparse row has too few finite neighbors")
)

func (s SparsePrecomputed) Validate() error {
	if s.Rows < 0 || s.Cols < 0 || s.Rows != s.Cols || len(s.IndPtr) != s.Rows+1 || len(s.Data) != len(s.Indices) {
		return fmt.Errorf("%w: inconsistent square CSR shape", ErrInvalidSparse)
	}
	if len(s.IndPtr) == 0 || s.IndPtr[0] != 0 || s.IndPtr[len(s.IndPtr)-1] != len(s.Data) {
		return fmt.Errorf("%w: invalid row pointers", ErrInvalidSparse)
	}
	for row := 0; row < s.Rows; row++ {
		if s.IndPtr[row] > s.IndPtr[row+1] {
			return fmt.Errorf("%w: row pointers are not monotonic", ErrInvalidSparse)
		}
		last := -1
		for p := s.IndPtr[row]; p < s.IndPtr[row+1]; p++ {
			col, value := s.Indices[p], s.Data[p]
			if col < 0 || col >= s.Cols || col <= last {
				return fmt.Errorf("%w: row %d indices must be sorted and unique", ErrInvalidSparse, row)
			}
			if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || (row == col && value != 0) {
				return fmt.Errorf("%w: invalid entry (%d,%d)", ErrInvalidSparse, row, col)
			}
			last = col
			q := sort.SearchInts(s.Indices[s.IndPtr[col]:s.IndPtr[col+1]], row) + s.IndPtr[col]
			if q >= s.IndPtr[col+1] || s.Indices[q] != row || s.Data[q] != value {
				return fmt.Errorf("%w: entry (%d,%d) is not symmetric", ErrInvalidSparse, row, col)
			}
		}
	}
	return nil
}

// ReferenceSparse runs HDBSCAN directly over CSR edges. It uses O(n+nnz)
// memory and returns ErrDisconnected with the component count when no spanning
// hierarchy exists. It never densifies the graph.
func ReferenceSparse(ctx context.Context, s SparsePrecomputed, cfg Config) (Result, error) {
	if err := s.Validate(); err != nil {
		return Result{}, err
	}
	if s.Rows < 2 {
		return Result{}, ErrTooFewPoints
	}
	if cfg.PredictionData {
		return Result{}, ErrPredictionUnsupported
	}
	if cfg.BranchDetectionData {
		return Result{}, ErrBranchUnsupported
	}
	cfg, err := normalizeConfig(cfg, s.Rows)
	if err != nil {
		return Result{}, err
	}
	k := min(cfg.MinSamples, s.Rows-1)
	core := make([]float64, s.Rows)
	maxDegree := 0
	for row := 0; row < s.Rows; row++ {
		if degree := s.IndPtr[row+1] - s.IndPtr[row]; degree > maxDegree {
			maxDegree = degree
		}
	}
	scratch := make([]float64, 0, maxDegree)
	for row := 0; row < s.Rows; row++ {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		values := scratch[:0]
		for p := s.IndPtr[row]; p < s.IndPtr[row+1]; p++ {
			if s.Indices[p] != row {
				values = append(values, s.Data[p])
			}
		}
		if len(values) < k {
			return Result{}, fmt.Errorf("%w: row %d has %d, needs %d", ErrInsufficientNeighbors, row, len(values), k)
		}
		sort.Float64s(values)
		core[row] = values[k-1]
	}
	edges := make([]MSTEdge, 0, len(s.Data)/2)
	for row := 0; row < s.Rows; row++ {
		for p := s.IndPtr[row]; p < s.IndPtr[row+1]; p++ {
			col := s.Indices[p]
			if col <= row {
				continue
			}
			d := s.Data[p] / cfg.Alpha
			if core[row] > d {
				d = core[row]
			}
			if core[col] > d {
				d = core[col]
			}
			edges = append(edges, MSTEdge{row, col, d})
		}
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].Distance != edges[j].Distance {
			return edges[i].Distance < edges[j].Distance
		}
		if edges[i].From != edges[j].From {
			return edges[i].From < edges[j].From
		}
		return edges[i].To < edges[j].To
	})
	uf := newMSTUF(s.Rows)
	mst := make([]MSTEdge, 0, s.Rows-1)
	for _, e := range edges {
		if uf.union(e.From, e.To) {
			mst = append(mst, e)
			if len(mst) == s.Rows-1 {
				break
			}
		}
	}
	if len(mst) != s.Rows-1 {
		roots := map[int]struct{}{}
		for i := 0; i < s.Rows; i++ {
			roots[uf.find(i)] = struct{}{}
		}
		return Result{}, fmt.Errorf("%w: %d components", ErrDisconnected, len(roots))
	}
	link := linkage(s.Rows, mst)
	condensed := condense(link, cfg.MinClusterSize)
	stability := stabilities(condensed)
	labels, probs, persistence := selectClusters(s.Rows, condensed, stability, cfg)
	return Result{Labels: labels, Probabilities: probs, ClusterPersistence: persistence, OutlierScores: outliers(s.Rows, condensed), MinimumSpanningTree: mst, SingleLinkageTree: link, CondensedTree: condensed, Metadata: Metadata{Algorithm: AlgorithmReference}, config: retainedConfig(cfg)}, nil
}

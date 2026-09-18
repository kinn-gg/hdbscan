package hdbscan

import (
	"errors"
	"fmt"
	"math"

	"github.com/kinn-gg/hdbscan/internal/distance"
)

// Metric computes the distance between equal-length vectors. Implementations
// must be safe for repeated calls. Built-in implementations do not allocate.
type Metric interface {
	Distance(a, b []float64) float64
}

// SquaredMetric additionally exposes an order-preserving squared distance.
// Algorithms use this method until an actual distance is semantically needed.
type SquaredMetric interface {
	Metric
	SquaredDistance(a, b []float64) float64
}

type builtinMetric uint8

const (
	SquaredEuclidean builtinMetric = iota
	Euclidean
	Manhattan
	Chebyshev
	Canberra
	BrayCurtis
)

func (m builtinMetric) Distance(a, b []float64) float64 {
	requireSameLength(a, b)
	switch m {
	case SquaredEuclidean:
		return distance.SquaredEuclidean(a, b)
	case Euclidean:
		return distance.Euclidean(a, b)
	case Manhattan:
		return distance.Manhattan(a, b)
	case Chebyshev:
		var result float64
		for i := range a {
			v := math.Abs(a[i] - b[i])
			if v > result {
				result = v
			}
		}
		return result
	case Canberra:
		var result float64
		for i := range a {
			denom := math.Abs(a[i]) + math.Abs(b[i])
			if denom != 0 {
				result += math.Abs(a[i]-b[i]) / denom
			}
		}
		return result
	case BrayCurtis:
		var numerator, denominator float64
		for i := range a {
			numerator += math.Abs(a[i] - b[i])
			denominator += math.Abs(a[i] + b[i])
		}
		if denominator == 0 {
			return 0
		}
		return numerator / denominator
	default:
		panic("hdbscan: unknown built-in metric")
	}
}

func (m builtinMetric) SquaredDistance(a, b []float64) float64 {
	requireSameLength(a, b)
	if m == Euclidean {
		return distance.SquaredEuclidean(a, b)
	}
	if m == SquaredEuclidean {
		d := distance.SquaredEuclidean(a, b)
		return d * d
	}
	value := m.Distance(a, b)
	return value * value
}

func (m builtinMetric) String() string {
	switch m {
	case SquaredEuclidean:
		return "sqeuclidean"
	case Euclidean:
		return "euclidean"
	case Manhattan:
		return "manhattan"
	case Chebyshev:
		return "chebyshev"
	case Canberra:
		return "canberra"
	case BrayCurtis:
		return "braycurtis"
	default:
		return "unknown"
	}
}

// SpatialIndex identifies an exact spatial index that is mathematically valid
// for a metric. The empty result means bounded brute force is required.
type SpatialIndex string

const (
	KDTreeIndex   SpatialIndex = "kdtree"
	BallTreeIndex SpatialIndex = "balltree"
)

// ValidSpatialIndexes reports optimized indexes that preserve a metric's
// geometry. The current engine implements KDTreeIndex for Euclidean only; the
// remaining declarations allow future selectors and external backends to avoid
// choosing an invalid index.
func ValidSpatialIndexes(metric Metric) []SpatialIndex {
	switch m := metric.(type) {
	case builtinMetric:
		switch m {
		case Euclidean:
			return []SpatialIndex{KDTreeIndex, BallTreeIndex}
		case Manhattan, Chebyshev:
			return []SpatialIndex{KDTreeIndex, BallTreeIndex}
		case Canberra, BrayCurtis:
			return []SpatialIndex{BallTreeIndex}
		}
	case MinkowskiMetric:
		return []SpatialIndex{KDTreeIndex, BallTreeIndex}
	}
	return nil
}

// MinkowskiMetric computes (sum(abs(a-b)^P))^(1/P). P must be finite and at
// least one. P=1 and P=2 use the optimized Manhattan and Euclidean kernels.
type MinkowskiMetric struct{ P float64 }

func NewMinkowski(p float64) (MinkowskiMetric, error) {
	if math.IsNaN(p) || math.IsInf(p, 0) || p < 1 {
		return MinkowskiMetric{}, fmt.Errorf("hdbscan: Minkowski p must be finite and >= 1")
	}
	return MinkowskiMetric{P: p}, nil
}

func (m MinkowskiMetric) Distance(a, b []float64) float64 {
	requireSameLength(a, b)
	if m.P < 1 || math.IsNaN(m.P) || math.IsInf(m.P, 0) {
		panic("hdbscan: invalid Minkowski metric")
	}
	return distance.Minkowski(a, b, m.P)
}
func (m MinkowskiMetric) String() string { return fmt.Sprintf("minkowski(%g)", m.P) }

// MetricFunc adapts a custom distance function. Custom functions may allocate.
type MetricFunc func(a, b []float64) float64

func (f MetricFunc) Distance(a, b []float64) float64 { requireSameLength(a, b); return f(a, b) }
func (MetricFunc) String() string                    { return "custom" }

func requireSameLength(a, b []float64) {
	if len(a) != len(b) {
		panic("hdbscan: distance vectors have different lengths")
	}
}

var ErrInvalidPrecomputed = errors.New("hdbscan: invalid precomputed distance matrix")

// Precomputed provides allocation-free indexed access to a square distance
// matrix. NaN, negative values, and nonzero diagonal entries are rejected.
// Positive infinity is accepted to represent disconnected pairs.
type Precomputed struct{ Dense64 }

func (p Precomputed) Validate() error {
	if err := p.Dense64.validateShape(); err != nil {
		return errors.Join(ErrInvalidPrecomputed, err)
	}
	if p.Rows != p.Cols {
		return fmt.Errorf("%w: matrix is %dx%d, want square", ErrInvalidPrecomputed, p.Rows, p.Cols)
	}
	for i, value := range p.Data {
		if math.IsNaN(value) || math.IsInf(value, -1) || value < 0 {
			return fmt.Errorf("%w: invalid value at data index %d", ErrInvalidPrecomputed, i)
		}
	}
	for i := 0; i < p.Rows; i++ {
		if p.Data[i*p.Cols+i] != 0 {
			return fmt.Errorf("%w: diagonal entry %d is not zero", ErrInvalidPrecomputed, i)
		}
		for j := 0; j < i; j++ {
			if p.Data[i*p.Cols+j] != p.Data[j*p.Cols+i] {
				return fmt.Errorf("%w: entries (%d,%d) and (%d,%d) differ", ErrInvalidPrecomputed, i, j, j, i)
			}
		}
	}
	return nil
}

func (p Precomputed) Distance(i, j int) float64 { return p.Data[i*p.Cols+j] }

package hdbscan

import (
	"errors"
	"fmt"
	"math"
)

// Dense64 is a contiguous, row-major matrix of float64 values. Data is not
// copied; callers must not mutate it while an algorithm is using the matrix.
type Dense64 struct {
	Data       []float64
	Rows, Cols int
}

var (
	ErrInvalidShape = errors.New("hdbscan: invalid dense matrix shape")
	ErrNonFinite    = errors.New("hdbscan: dense matrix contains a non-finite value")
)

// Validate checks the shape, backing storage length, and every value. Empty
// matrices are valid when their backing storage is empty.
func (m Dense64) Validate() error {
	if err := m.validateShape(); err != nil {
		return err
	}
	for i, value := range m.Data {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("%w at data index %d", ErrNonFinite, i)
		}
	}
	return nil
}

func (m Dense64) validateShape() error {
	if m.Rows < 0 || m.Cols < 0 {
		return fmt.Errorf("%w: negative dimensions %dx%d", ErrInvalidShape, m.Rows, m.Cols)
	}
	// Division avoids overflowing Rows*Cols.
	if m.Rows != 0 && m.Cols > int(^uint(0)>>1)/m.Rows {
		return fmt.Errorf("%w: dimensions %dx%d overflow int", ErrInvalidShape, m.Rows, m.Cols)
	}
	want := m.Rows * m.Cols
	if len(m.Data) != want {
		return fmt.Errorf("%w: data length %d, want %d for %dx%d", ErrInvalidShape, len(m.Data), want, m.Rows, m.Cols)
	}
	return nil
}

// Row returns a view of row i. It panics when i is out of range, like a slice
// index expression. The returned slice aliases Data.
func (m Dense64) Row(i int) []float64 {
	if uint(i) >= uint(m.Rows) {
		panic("hdbscan: row index out of range")
	}
	start := i * m.Cols
	return m.Data[start : start+m.Cols : start+m.Cols]
}

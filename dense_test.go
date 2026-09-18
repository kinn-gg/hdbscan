package hdbscan

import (
	"errors"
	"math"
	"testing"
)

func TestDense64Validate(t *testing.T) {
	tests := []struct {
		name   string
		matrix Dense64
		want   error
	}{
		{"empty", Dense64{}, nil},
		{"zero columns", Dense64{Rows: 2}, nil},
		{"valid", Dense64{Data: []float64{1, 2, 3, 4}, Rows: 2, Cols: 2}, nil},
		{"short", Dense64{Data: []float64{1}, Rows: 1, Cols: 2}, ErrInvalidShape},
		{"negative", Dense64{Rows: -1}, ErrInvalidShape},
		{"overflow", Dense64{Rows: int(^uint(0) >> 1), Cols: 2}, ErrInvalidShape},
		{"nan", Dense64{Data: []float64{math.NaN()}, Rows: 1, Cols: 1}, ErrNonFinite},
		{"infinity", Dense64{Data: []float64{math.Inf(1)}, Rows: 1, Cols: 1}, ErrNonFinite},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.matrix.Validate()
			if !errors.Is(err, test.want) {
				t.Fatalf("Validate() error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestDense64RowIsBoundedView(t *testing.T) {
	m := Dense64{Data: []float64{1, 2, 3, 4}, Rows: 2, Cols: 2}
	row := m.Row(1)
	if row[0] != 3 || row[1] != 4 || cap(row) != 2 {
		t.Fatalf("Row(1) = %v cap %d", row, cap(row))
	}
}

func TestPrecomputedValidationAndAccess(t *testing.T) {
	p := Precomputed{Dense64{Data: []float64{0, math.Inf(1), math.Inf(1), 0}, Rows: 2, Cols: 2}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(p.Distance(0, 1), 1) {
		t.Fatal("disconnected distance was not preserved")
	}
	bad := Precomputed{Dense64{Data: []float64{1}, Rows: 1, Cols: 1}}
	if !errors.Is(bad.Validate(), ErrInvalidPrecomputed) {
		t.Fatal("nonzero diagonal accepted")
	}
}

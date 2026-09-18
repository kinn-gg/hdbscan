package distance

// Kernels is a dispatch table selected once by an algorithm before entering
// its hot loops. Keeping function selection here avoids architecture checks for
// every vector pair.
type Kernels struct {
	Name                  string
	Dot                   func(a, b []float64) float64
	SquaredEuclidean      func(a, b []float64) float64
	SquaredEuclideanBlock func(dst, a, b []float64, aRows, bRows, dims int)
}

var scalarKernels = Kernels{
	Name:                  "scalar",
	Dot:                   Dot,
	SquaredEuclidean:      SquaredEuclidean,
	SquaredEuclideanBlock: SquaredEuclideanBlock,
}

// Select returns the best measured kernel table for this process. M1 keeps the
// portable scalar implementation as the default on every architecture; tuned
// tables can be added here without changing algorithm hot loops.
func Select() Kernels { return scalarKernels }

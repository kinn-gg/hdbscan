//go:build arm64

package distance

const arm64VectorThreshold = 64

var arm64VectorKernels = Kernels{
	Name:                  "neon",
	Dot:                   dotNEON,
	SquaredEuclidean:      squaredEuclideanNEON,
	SquaredEuclideanBlock: squaredEuclideanBlockNEON,
}

func selectKernels(dims int) Kernels {
	if dims < arm64VectorThreshold {
		return scalarKernels
	}
	return arm64VectorKernels
}

func dotNEON(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return dotVector(&a[0], &b[0], len(a))
}

func squaredEuclideanNEON(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return squaredEuclideanVector(&a[0], &b[0], len(a))
}

func squaredEuclideanBlockNEON(dst, a, b []float64, aRows, bRows, dims int) {
	squaredEuclideanBlock(dst, a, b, aRows, bRows, dims, squaredEuclideanNEON)
}

//go:noescape
func dotVector(a, b *float64, n int) float64

//go:noescape
func squaredEuclideanVector(a, b *float64, n int) float64

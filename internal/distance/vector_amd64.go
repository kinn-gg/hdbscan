//go:build amd64 && amd64.v3

package distance

const amd64VectorThreshold = 4

var amd64VectorKernels = Kernels{
	Name:                  "avx2",
	Dot:                   dotAVX2,
	SquaredEuclidean:      squaredEuclideanAVX2,
	SquaredEuclideanBlock: squaredEuclideanBlockAVX2,
}

func selectKernels(dims int) Kernels {
	if dims < amd64VectorThreshold {
		return scalarKernels
	}
	return amd64VectorKernels
}

func dotAVX2(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return dotVector(&a[0], &b[0], len(a))
}

func squaredEuclideanAVX2(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return squaredEuclideanVector(&a[0], &b[0], len(a))
}

func squaredEuclideanBlockAVX2(dst, a, b []float64, aRows, bRows, dims int) {
	squaredEuclideanBlock(dst, a, b, aRows, bRows, dims, squaredEuclideanAVX2)
}

//go:noescape
func dotVector(a, b *float64, n int) float64

//go:noescape
func squaredEuclideanVector(a, b *float64, n int) float64

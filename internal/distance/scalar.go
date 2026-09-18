// Package distance implements allocation-free vector distance kernels.
package distance

import "math"

func SquaredEuclidean(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	_ = b[len(a)-1]
	var sum float64
	for i, av := range a {
		d := av - b[i]
		sum += d * d
	}
	return sum
}

// Euclidean uses a scaled sum of squares so representable norms do not
// overflow or underflow merely because their square is not representable.
func Euclidean(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	_ = b[len(a)-1]
	var scale, sumsq float64
	for i, av := range a {
		x := math.Abs(av - b[i])
		if x == 0 {
			continue
		}
		if scale < x {
			r := scale / x
			sumsq = 1 + sumsq*r*r
			scale = x
		} else {
			r := x / scale
			sumsq += r * r
		}
	}
	return scale * math.Sqrt(sumsq)
}

func Manhattan(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	_ = b[len(a)-1]
	var sum float64
	for i, av := range a {
		sum += math.Abs(av - b[i])
	}
	return sum
}

func Minkowski(a, b []float64, p float64) float64 {
	if len(a) == 0 {
		return 0
	}
	if p == 1 {
		return Manhattan(a, b)
	}
	if p == 2 {
		return Euclidean(a, b)
	}
	_ = b[len(a)-1]
	var scale float64
	for i, av := range a {
		x := math.Abs(av - b[i])
		if x > scale {
			scale = x
		}
	}
	if scale == 0 {
		return 0
	}
	var sum float64
	for i, av := range a {
		sum += math.Pow(math.Abs(av-b[i])/scale, p)
	}
	return scale * math.Pow(sum, 1/p)
}

func Dot(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	_ = b[len(a)-1]
	var sum float64
	for i, av := range a {
		sum += av * b[i]
	}
	return sum
}

// SquaredEuclideanBlock fills dst with distances between each row in a and b.
// dst is row-major with aRows*bRows entries. Inputs and dst must not overlap.
func SquaredEuclideanBlock(dst, a, b []float64, aRows, bRows, dims int) {
	for i := 0; i < aRows; i++ {
		arow := a[i*dims : (i+1)*dims]
		for j := 0; j < bRows; j++ {
			dst[i*bRows+j] = SquaredEuclidean(arow, b[j*dims:(j+1)*dims])
		}
	}
}

//go:build amd64 && amd64.v3

package distance

import "testing"

func vectorKernelsForTest() Kernels { return amd64VectorKernels }

func TestAMD64MeasuredThreshold(t *testing.T) {
	if got := Select(3).Name; got != "scalar" {
		t.Fatalf("Select(3) = %q", got)
	}
	if got := Select(4).Name; got != "avx2" {
		t.Fatalf("Select(4) = %q", got)
	}
}

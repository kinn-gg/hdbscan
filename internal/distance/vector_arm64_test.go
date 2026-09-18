//go:build arm64

package distance

import "testing"

func vectorKernelsForTest() Kernels { return arm64VectorKernels }

func TestARM64MeasuredThreshold(t *testing.T) {
	if got := Select(63).Name; got != "scalar" {
		t.Fatalf("Select(63) = %q", got)
	}
	if got := Select(64).Name; got != "neon" {
		t.Fatalf("Select(64) = %q", got)
	}
}

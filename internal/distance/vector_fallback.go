//go:build !arm64 && (!amd64 || !amd64.v3)

package distance

func selectKernels(_ int) Kernels { return scalarKernels }

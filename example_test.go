package hdbscan_test

import (
	"context"
	"fmt"
	"io"

	"github.com/kinn-gg/hdbscan"
)

func ExampleFitPredict() {
	points := hdbscan.Dense64{Data: []float64{0, 0, 0, .1, 10, 10, 10, 10.1}, Rows: 4, Cols: 2}
	labels, err := hdbscan.FitPredict(context.Background(), points, hdbscan.Config{MinClusterSize: 2, MinSamples: 1})
	if err != nil {
		panic(err)
	}
	fmt.Println(len(labels))
	// Output: 4
}

func ExampleResult_WriteCondensedTree() {
	points := hdbscan.Dense64{Data: []float64{0, 0, 1, 1}, Rows: 2, Cols: 2}
	result, err := hdbscan.Fit(context.Background(), points, hdbscan.Config{MinClusterSize: 2, MinSamples: 1, AllowSingleCluster: true})
	if err != nil {
		panic(err)
	}
	if err := result.WriteCondensedTree(io.Discard, hdbscan.ExportJSON); err != nil {
		panic(err)
	}
	fmt.Println("encoded")
	// Output: encoded
}

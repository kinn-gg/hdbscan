package hdbscan

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestPublicModelAPIAndOwnership(t *testing.T) {
	x := randomDense(40, 3, 91)
	model, err := New(Config{MinClusterSize: 3, MinSamples: 2, Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Fit(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	labels, err := model.FitPredict(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	labels[0] = 999
	if result.Labels[0] == 999 {
		t.Fatal("FitPredict aliases Result labels")
	}
	flat, err := result.Extract(ExtractionConfig{ClusterSelectionMethod: Leaf})
	if err != nil {
		t.Fatal(err)
	}
	flat.Labels[0] = 999
	if result.Labels[0] == 999 {
		t.Fatal("Extract aliases Result labels")
	}
}

func TestNewValidationAndCancellation(t *testing.T) {
	if _, err := New(Config{MinClusterSize: 1}); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("error = %v", err)
	}
	model, err := New(Config{MinClusterSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := model.Fit(ctx, randomDense(10, 2, 1)); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v", err)
	}
}

func TestStreamingExports(t *testing.T) {
	result, err := Fit(context.Background(), randomDense(12, 2, 7), Config{MinClusterSize: 2, MinSamples: 2})
	if err != nil {
		t.Fatal(err)
	}
	var csv bytes.Buffer
	if err := result.WriteMST(&csv, ExportCSV); err != nil {
		t.Fatal(err)
	}
	if lines := strings.Count(csv.String(), "\n"); lines != len(result.MinimumSpanningTree)+1 {
		t.Fatalf("CSV lines = %d", lines)
	}
	var stream bytes.Buffer
	if err := result.WriteCondensedTree(&stream, ExportJSON); err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(&stream)
	for range result.CondensedTree {
		var edge CondensedEdge
		if err := decoder.Decode(&edge); err != nil {
			t.Fatal(err)
		}
	}
	if err := result.WriteMST(&stream, ExportFormat("xml")); err == nil {
		t.Fatal("expected format error")
	}
}

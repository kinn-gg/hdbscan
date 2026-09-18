package hdbscan

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"testing"
)

func predictionFixture() Dense64 {
	return Dense64{Data: []float64{
		-5.2, -5.1, -5.0, -4.9, -4.8, -5.1, -5.1, -4.8, -4.9, -5.2,
		4.8, 5.1, 5.0, 4.9, 5.2, 5.1, 4.9, 4.8, 5.1, 5.2,
	}, Rows: 10, Cols: 2}
}

func TestPredictionIsOptInAndOwnsTrainingData(t *testing.T) {
	ctx := context.Background()
	x := predictionFixture()
	plain, err := Fit(ctx, x, Config{MinClusterSize: 2, MinSamples: 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := plain.ApproximatePredict(ctx, Dense64{Data: []float64{-5, -5}, Rows: 1, Cols: 2}); !errors.Is(err, ErrPredictionData) {
		t.Fatalf("error = %v", err)
	}

	result, err := Fit(ctx, x, Config{MinClusterSize: 2, MinSamples: 2, PredictionData: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := range x.Data {
		x.Data[i] = 1000
	}
	labels, strengths, err := result.ApproximatePredict(ctx, Dense64{Data: []float64{-5, -5, 5, 5}, Rows: 2, Cols: 2})
	if err != nil {
		t.Fatal(err)
	}
	if labels[0] < 0 || labels[1] < 0 || labels[0] == labels[1] {
		t.Fatalf("labels = %v", labels)
	}
	for _, v := range strengths {
		if v <= 0 || v > 1 {
			t.Fatalf("strength = %g", v)
		}
	}
}

func TestPredictionMembershipAndScores(t *testing.T) {
	ctx := context.Background()
	result, err := Fit(ctx, predictionFixture(), Config{MinClusterSize: 2, MinSamples: 2, PredictionData: true})
	if err != nil {
		t.Fatal(err)
	}
	points := Dense64{Data: []float64{-5, -5, 5, 5, 50, 50}, Rows: 3, Cols: 2}
	labels, _, err := result.ApproximatePredict(ctx, points)
	if err != nil {
		t.Fatal(err)
	}
	vectors, err := result.MembershipVectors(ctx, points)
	if err != nil {
		t.Fatal(err)
	}
	if vectors.Rows != 3 || vectors.Cols != result.ClusterCount() {
		t.Fatalf("shape = %dx%d", vectors.Rows, vectors.Cols)
	}
	for row := 0; row < vectors.Rows; row++ {
		sum, best := 0.0, 0
		for col, value := range vectors.Row(row) {
			if value < 0 || value > 1 || math.IsNaN(value) {
				t.Fatalf("membership = %g", value)
			}
			sum += value
			if value > vectors.Row(row)[best] {
				best = col
			}
		}
		if sum > 1+1e-12 {
			t.Fatalf("row sum = %g", sum)
		}
		if row < 2 && best != labels[row] {
			t.Fatalf("row %d argmax=%d label=%d", row, best, labels[row])
		}
	}
	scores, err := result.ApproximatePredictScores(ctx, points)
	if err != nil {
		t.Fatal(err)
	}
	for _, score := range scores {
		if math.IsNaN(score) || score > 1 {
			t.Fatalf("score = %g", score)
		}
	}
	all, err := result.AllPointsMembershipVectors(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if all.Rows != predictionFixture().Rows || all.Cols != result.ClusterCount() {
		t.Fatalf("all shape = %dx%d", all.Rows, all.Cols)
	}
}

func TestPredictionPinnedUpstreamFixture(t *testing.T) {
	b, err := os.ReadFile("testdata/parity/synthetic_blobs.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture parityCase
	if err := json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	x := Dense64{Data: decodeNums(fixture.Case.Data), Rows: fixture.Case.Rows, Cols: fixture.Case.Cols}
	result, err := Fit(context.Background(), x, Config{MinClusterSize: 5, MinSamples: 5, PredictionData: true, Algorithm: AlgorithmReference})
	if err != nil {
		t.Fatal(err)
	}
	q := Dense64{Data: []float64{4.5, -1, 5.7, 3.1, -6.4, 7.7, 0, 0}, Rows: 4, Cols: 2}
	wantLabels := []int{1, 2, 0, -1}
	wantStrengths := []float64{1, 0.6751027115642216, 0.7224072621432686, 0}
	wantScores := []float64{-0.06483954058634919, 0.32489728843577836, 0.27759273785673144, 0.9521490296534547}
	wantMemberships := []float64{4.080319370440225e-35, 0.9746176409022883, 7.842487890953671e-35, 0.0011936367705573288, 0.002175541113635202, 0.6717335326028261, 0.7217776246357392, 0.0003097526473338171, 0.00031988358755448475, 0.011116848637371383, 0.017884462832701876, 0.014793823343511387}
	labels, strengths, err := result.ApproximatePredict(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	scores, err := result.ApproximatePredictScores(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	memberships, err := result.MembershipVectors(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	for i := range wantLabels {
		if labels[i] != wantLabels[i] {
			t.Errorf("label[%d]=%d want %d", i, labels[i], wantLabels[i])
		}
		closePrediction(t, "strength", i, strengths[i], wantStrengths[i])
		closePrediction(t, "score", i, scores[i], wantScores[i])
	}
	for i, want := range wantMemberships {
		closePrediction(t, "membership", i, memberships.Data[i], want)
	}
	all, err := result.AllPointsMembershipVectors(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	wantAll := map[int][]float64{
		1:  {0.014564607143082112, 0.04515573143854055, 0.6270760734553907},
		2:  {0.965065598529661, 0.0029015257672342603, 0.0031174748585587296},
		20: {0.9704707407204425, 0.0024041542661024824, 0.002572821228459363},
		40: {0.008332551537621479, 0.023906530692483587, 0.5920616969941077},
	}
	// The pinned Cython oracle leaves underflowed off-cluster cells uninitialized;
	// rows with material values provide a stable parity contract.
	for row, want := range wantAll {
		for col, value := range want {
			closePrediction(t, "all-membership", row*all.Cols+col, all.Row(row)[col], value)
		}
	}
}

func closePrediction(t *testing.T, name string, i int, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-7+1e-6*math.Abs(want) {
		t.Errorf("%s[%d]=%.15g want %.15g", name, i, got, want)
	}
}

func TestPredictionEdgeCasesAndBuffers(t *testing.T) {
	ctx := context.Background()
	result, err := Fit(ctx, predictionFixture(), Config{MinClusterSize: 20, MinSamples: 2, PredictionData: true})
	if err != nil {
		t.Fatal(err)
	}
	point := Dense64{Data: []float64{0, 0}, Rows: 1, Cols: 2}
	labels, strengths, err := result.ApproximatePredict(ctx, point)
	if err != nil {
		t.Fatal(err)
	}
	if labels[0] != -1 || strengths[0] != 0 || result.ClusterCount() != 0 {
		t.Fatalf("got %v %v clusters=%d", labels, strengths, result.ClusterCount())
	}
	if err := result.PredictInto(ctx, point, nil, nil); err == nil {
		t.Fatal("expected buffer error")
	}
	if _, _, err := result.ApproximatePredict(ctx, Dense64{Data: []float64{1}, Rows: 1, Cols: 1}); !errors.Is(err, ErrInvalidShape) {
		t.Fatalf("error = %v", err)
	}
	if _, _, err := result.ApproximatePredict(ctx, Dense64{Data: []float64{math.NaN(), 0}, Rows: 1, Cols: 2}); !errors.Is(err, ErrNonFinite) {
		t.Fatalf("error = %v", err)
	}
	pre := Precomputed{Dense64: Dense64{Data: []float64{0, 1, 1, 0}, Rows: 2, Cols: 2}}
	if _, err := ReferencePrecomputed(ctx, pre, Config{MinClusterSize: 2, PredictionData: true}); !errors.Is(err, ErrPredictionUnsupported) {
		t.Fatalf("error = %v", err)
	}
}

func TestPredictionAllowedSingleCluster(t *testing.T) {
	result, err := Fit(context.Background(), predictionFixture(), Config{MinClusterSize: 8, MinSamples: 2, AllowSingleCluster: true, PredictionData: true})
	if err != nil {
		t.Fatal(err)
	}
	if result.ClusterCount() != 1 {
		t.Fatalf("cluster count = %d", result.ClusterCount())
	}
	v, err := result.MembershipVector(context.Background(), []float64{-5, -5})
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 1 || v[0] < 0 || v[0] > 1 {
		t.Fatalf("membership = %v", v)
	}
}

func TestPredictionCallerBuffersHaveBoundedAllocations(t *testing.T) {
	result, err := Fit(context.Background(), predictionFixture(), Config{MinClusterSize: 2, MinSamples: 2, PredictionData: true})
	if err != nil {
		t.Fatal(err)
	}
	points := predictionFixture()
	labels, strengths := make([]int, points.Rows), make([]float64, points.Rows)
	a := testing.AllocsPerRun(20, func() {
		if err := result.PredictInto(context.Background(), points, labels, strengths); err != nil {
			panic(err)
		}
	})
	if a > 8 {
		t.Fatalf("allocations per batch = %g", a)
	}
	one := Dense64{Data: points.Data[:points.Cols], Rows: 1, Cols: points.Cols}
	oneLabels, oneStrength := make([]int, 1), make([]float64, 1)
	oneAllocs := testing.AllocsPerRun(20, func() {
		if err := result.PredictInto(context.Background(), one, oneLabels, oneStrength); err != nil {
			panic(err)
		}
	})
	if a > oneAllocs+1 {
		t.Fatalf("allocations scale with points: one=%g batch=%g", oneAllocs, a)
	}
}

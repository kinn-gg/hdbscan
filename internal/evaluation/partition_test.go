package evaluation

import (
	"math"
	"slices"
	"testing"
)

func TestCanonicalLabels(t *testing.T) {
	want := []int{0, -1, 1, 0, 2, 1}
	got := CanonicalLabels([]int{42, -1, 7, 42, 99, 7})
	if !slices.Equal(got, want) {
		t.Fatalf("CanonicalLabels() = %v, want %v", got, want)
	}
}

func TestPartitionComparisons(t *testing.T) {
	a := []int{4, 4, -1, 9, 9, 9}
	b := []int{0, 0, -1, 1, 1, 1}
	c := []int{0, 1, -1, 1, 1, 1}
	if !SamePartition(a, b) || SamePartition(a, c) {
		t.Fatal("partition equivalence mismatch")
	}
	if got := AdjustedRandIndex(a, b); got != 1 {
		t.Fatalf("ARI(equal) = %v, want 1", got)
	}
	if got := AdjustedRandIndex(a, c); math.IsNaN(got) || got >= 1 {
		t.Fatalf("ARI(different) = %v, want finite value below 1", got)
	}
}

func FuzzCanonicalLabels(f *testing.F) {
	f.Add([]byte{1, 1, 255, 7, 3, 7})
	f.Fuzz(func(t *testing.T, raw []byte) {
		labels := make([]int, len(raw))
		for i, value := range raw {
			if value == 255 {
				labels[i] = -1
			} else {
				labels[i] = int(value)
			}
		}
		once := CanonicalLabels(labels)
		twice := CanonicalLabels(once)
		if !slices.Equal(once, twice) {
			t.Fatalf("canonicalization is not idempotent: %v then %v", once, twice)
		}
		if !SamePartition(labels, once) {
			t.Fatalf("canonicalization changed partition: %v to %v", labels, once)
		}
	})
}

func BenchmarkCanonicalLabels(b *testing.B) {
	labels := make([]int, 100_000)
	for i := range labels {
		if i%13 == 0 {
			labels[i] = -1
		} else {
			labels[i] = (i / 17) % 100
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = CanonicalLabels(labels)
	}
}

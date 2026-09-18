// Package evaluation contains comparison helpers for parity and benchmark tests.
package evaluation

// CanonicalLabels renumbers non-noise clusters by first appearance. Noise remains
// -1. The returned slice never aliases labels.
func CanonicalLabels(labels []int) []int {
	canonical := make([]int, len(labels))
	next := 0
	seen := make(map[int]int)
	for i, label := range labels {
		if label == -1 {
			canonical[i] = -1
			continue
		}
		mapped, ok := seen[label]
		if !ok {
			mapped = next
			seen[label] = mapped
			next++
		}
		canonical[i] = mapped
	}
	return canonical
}

// SamePartition reports whether two labelings have identical pairwise cluster
// membership. Noise is treated as a label: all noise points are co-members.
func SamePartition(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		for j := 0; j < i; j++ {
			if (a[i] == a[j]) != (b[i] == b[j]) {
				return false
			}
		}
	}
	return true
}

// AdjustedRandIndex returns the adjusted Rand index of two partitions. It returns
// 1 for two equal partitions with fewer than two observations or when both
// partitions have zero expected and observed pair counts.
func AdjustedRandIndex(a, b []int) float64 {
	if len(a) != len(b) {
		return 0
	}
	if len(a) < 2 {
		return 1
	}

	type pair struct{ a, b int }
	joint := make(map[pair]int)
	countA := make(map[int]int)
	countB := make(map[int]int)
	for i := range a {
		joint[pair{a[i], b[i]}]++
		countA[a[i]]++
		countB[b[i]]++
	}

	choose2 := func(n int) float64 { return float64(n*(n-1)) / 2 }
	var sumJoint, sumA, sumB float64
	for _, count := range joint {
		sumJoint += choose2(count)
	}
	for _, count := range countA {
		sumA += choose2(count)
	}
	for _, count := range countB {
		sumB += choose2(count)
	}
	total := choose2(len(a))
	expected := sumA * sumB / total
	maximum := (sumA + sumB) / 2
	if maximum == expected {
		if SamePartition(a, b) {
			return 1
		}
		return 0
	}
	return (sumJoint - expected) / (maximum - expected)
}

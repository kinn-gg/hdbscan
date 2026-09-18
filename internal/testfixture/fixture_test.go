package testfixture

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"sort"
	"testing"
)

func TestCommittedFixtures(t *testing.T) {
	fixtures, err := LoadAll(os.DirFS("../../testdata/parity"))
	if err != nil {
		t.Fatal(err)
	}
	covered := make([]string, 0, len(fixtures))
	for _, fixture := range fixtures {
		covered = append(covered, fixture.Case.Name)
	}
	want := []string{"all_noise", "allow_single_cluster", "alpha_nondefault", "branch_shapes", "duplicates_ties_eom", "duplicates_ties_leaf", "empty", "high_dimensional", "max_cluster_size", "metric_braycurtis", "metric_canberra", "metric_chebyshev", "metric_manhattan", "metric_minkowski_p3", "nonfinite_rows", "precomputed_dense_connected", "precomputed_dense_missing_edges", "precomputed_disconnected", "precomputed_sparse_connected", "selection_epsilon", "selection_persistence", "singleton", "synthetic_blobs"}
	sort.Strings(covered)
	if len(covered) != len(want) {
		t.Fatalf("fixture names=%v want %v", covered, want)
	}
	for i := range want {
		if covered[i] != want[i] {
			t.Fatalf("fixture names=%v want %v", covered, want)
		}
	}

	manifestData, err := os.ReadFile("../../testdata/parity/SHA256SUMS.json")
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]string
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatal(err)
	}
	if len(manifest) != len(want) {
		t.Fatalf("checksum entries=%d want %d", len(manifest), len(want))
	}
	for _, name := range want {
		filename := name + ".json"
		data, err := os.ReadFile("../../testdata/parity/" + filename)
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(data)
		if got := hex.EncodeToString(digest[:]); got != manifest[filename] {
			t.Errorf("%s checksum=%s want %s", filename, got, manifest[filename])
		}
	}
}

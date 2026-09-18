package testfixture

import (
	"os"
	"testing"
)

func TestCommittedFixtures(t *testing.T) {
	fixtures, err := LoadAll(os.DirFS("../../testdata/parity"))
	if err != nil {
		t.Fatal(err)
	}
	if len(fixtures) < 10 {
		t.Fatalf("loaded %d fixtures, want at least 10", len(fixtures))
	}
	covered := make(map[string]bool)
	for _, fixture := range fixtures {
		covered[fixture.Case.Name] = true
	}
	for _, name := range []string{"empty", "singleton", "duplicates_ties_eom", "all_noise", "allow_single_cluster", "nonfinite_rows", "high_dimensional", "synthetic_blobs", "precomputed_disconnected"} {
		if !covered[name] {
			t.Errorf("required fixture %q is missing", name)
		}
	}
}

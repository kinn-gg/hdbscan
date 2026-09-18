package benchmarkdata

import (
	"os"
	"slices"
	"testing"
)

func loadManifest(t testing.TB) Manifest {
	t.Helper()
	file, err := os.Open("../../testdata/benchmarks/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	manifest, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}

func TestManifestCoverageAndDeterminism(t *testing.T) {
	manifest := loadManifest(t)
	if len(manifest.Recipes) != 12 {
		t.Fatalf("got %d recipes, want 12", len(manifest.Recipes))
	}
	categories := make(map[string]map[string]bool)
	for _, recipe := range manifest.Recipes {
		size := recipe.Name[len(recipe.Name)-5:]
		if categories[recipe.Generator] == nil {
			categories[recipe.Generator] = make(map[string]bool)
		}
		categories[recipe.Generator][size] = true

		small := recipe
		if small.Rows > 128 {
			small.Rows = 128
		}
		first, err := Generate(small)
		if err != nil {
			t.Fatal(err)
		}
		second, err := Generate(small)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(first, second) {
			t.Fatalf("recipe %q is not deterministic", recipe.Name)
		}
	}
	for _, generator := range []string{"gaussian_blobs", "duplicate_grid", "noise_mixture"} {
		if len(categories[generator]) < 3 {
			t.Errorf("generator %q does not cover small, medium, and large", generator)
		}
	}
}

func BenchmarkCorpusGeneration(b *testing.B) {
	manifest := loadManifest(b)
	for _, recipe := range manifest.Recipes {
		if recipe.Rows > 10_000 {
			continue
		}
		b.Run(recipe.Name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(recipe.Rows * recipe.Dimensions * 8))
			for b.Loop() {
				if _, err := Generate(recipe); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

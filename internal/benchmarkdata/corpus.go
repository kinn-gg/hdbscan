// Package benchmarkdata generates deterministic datasets from the benchmark manifest.
package benchmarkdata

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
)

type Manifest struct {
	SchemaVersion int      `json:"schema_version"`
	Recipes       []Recipe `json:"recipes"`
}

type Recipe struct {
	Name              string  `json:"name"`
	Generator         string  `json:"generator"`
	Seed              uint64  `json:"seed"`
	Rows              int     `json:"rows"`
	Dimensions        int     `json:"dimensions"`
	Clusters          int     `json:"clusters,omitempty"`
	Spread            float64 `json:"spread,omitempty"`
	NoiseFraction     float64 `json:"noise_fraction,omitempty"`
	DuplicateFraction float64 `json:"duplicate_fraction,omitempty"`
}

func Load(reader io.Reader) (Manifest, error) {
	var manifest Manifest
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil {
		return manifest, err
	}
	if manifest.SchemaVersion != 1 {
		return manifest, fmt.Errorf("benchmark schema version %d, want 1", manifest.SchemaVersion)
	}
	seen := make(map[string]bool)
	for _, recipe := range manifest.Recipes {
		if err := recipe.Validate(); err != nil {
			return manifest, fmt.Errorf("recipe %q: %w", recipe.Name, err)
		}
		if seen[recipe.Name] {
			return manifest, fmt.Errorf("duplicate recipe %q", recipe.Name)
		}
		seen[recipe.Name] = true
	}
	return manifest, nil
}

func (recipe Recipe) Validate() error {
	if recipe.Name == "" || recipe.Rows <= 0 || recipe.Dimensions <= 0 {
		return fmt.Errorf("name, rows, and dimensions are required")
	}
	switch recipe.Generator {
	case "gaussian_blobs":
		if recipe.Clusters <= 0 || recipe.Spread <= 0 {
			return fmt.Errorf("gaussian_blobs requires positive clusters and spread")
		}
	case "duplicate_grid":
		if recipe.Dimensions != 2 || recipe.DuplicateFraction < 0 || recipe.DuplicateFraction > 1 {
			return fmt.Errorf("duplicate_grid requires 2 dimensions and duplicate_fraction in [0,1]")
		}
	case "noise_mixture":
		if recipe.Clusters <= 0 || recipe.Spread <= 0 || recipe.NoiseFraction < 0 || recipe.NoiseFraction > 1 {
			return fmt.Errorf("noise_mixture parameters are invalid")
		}
	default:
		return fmt.Errorf("unknown generator %q", recipe.Generator)
	}
	return nil
}

// Generate returns row-major float64 observations. It uses a private PRNG so its
// output does not change with Go's math/rand implementation.
func Generate(recipe Recipe) ([]float64, error) {
	if err := recipe.Validate(); err != nil {
		return nil, err
	}
	rng := splitmix64{state: recipe.Seed}
	data := make([]float64, recipe.Rows*recipe.Dimensions)
	switch recipe.Generator {
	case "gaussian_blobs":
		fillBlobs(data, recipe, &rng, 0)
	case "noise_mixture":
		noiseRows := int(float64(recipe.Rows) * recipe.NoiseFraction)
		for i := 0; i < noiseRows*recipe.Dimensions; i++ {
			data[i] = rng.uniform()*40 - 20
		}
		clustered := recipe
		clustered.Rows -= noiseRows
		fillBlobs(data[noiseRows*recipe.Dimensions:], clustered, &rng, 0)
	case "duplicate_grid":
		unique := recipe.Rows - int(float64(recipe.Rows)*recipe.DuplicateFraction)
		if unique < 1 {
			unique = 1
		}
		width := int(math.Ceil(math.Sqrt(float64(unique))))
		for row := 0; row < recipe.Rows; row++ {
			point := row % unique
			data[row*2] = float64(point % width)
			data[row*2+1] = float64(point / width)
		}
	}
	return data, nil
}

func fillBlobs(data []float64, recipe Recipe, rng *splitmix64, offset int) {
	centers := make([]float64, recipe.Clusters*recipe.Dimensions)
	for i := range centers {
		centers[i] = rng.uniform()*16 - 8
	}
	for row := 0; row < recipe.Rows; row++ {
		cluster := (row + offset) % recipe.Clusters
		for column := 0; column < recipe.Dimensions; column++ {
			data[row*recipe.Dimensions+column] = centers[cluster*recipe.Dimensions+column] + rng.normal()*recipe.Spread
		}
	}
}

type splitmix64 struct {
	state    uint64
	spare    float64
	hasSpare bool
}

func (rng *splitmix64) next() uint64 {
	rng.state += 0x9e3779b97f4a7c15
	z := rng.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (rng *splitmix64) uniform() float64 {
	return float64(rng.next()>>11) * (1.0 / (1 << 53))
}

func (rng *splitmix64) normal() float64 {
	if rng.hasSpare {
		rng.hasSpare = false
		return rng.spare
	}
	u1 := rng.uniform()
	if u1 == 0 {
		u1 = math.SmallestNonzeroFloat64
	}
	u2 := rng.uniform()
	radius := math.Sqrt(-2 * math.Log(u1))
	angle := 2 * math.Pi * u2
	rng.spare = radius * math.Sin(angle)
	rng.hasSpare = true
	return radius * math.Cos(angle)
}

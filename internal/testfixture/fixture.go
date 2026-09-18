// Package testfixture reads the versioned Python parity fixture format.
package testfixture

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"path"
	"strings"
)

const SchemaVersion = 1
const UpstreamVersion = "0.8.44"
const UpstreamCommit = "dfdc9ca2b265ab6d50cf64b428199785700af240"

type Float64 float64

func (value *Float64) UnmarshalJSON(data []byte) error {
	if bytes.Equal(data, []byte(`"NaN"`)) {
		*value = Float64(math.NaN())
		return nil
	}
	if bytes.Equal(data, []byte(`"+Inf"`)) {
		*value = Float64(math.Inf(1))
		return nil
	}
	if bytes.Equal(data, []byte(`"-Inf"`)) {
		*value = Float64(math.Inf(-1))
		return nil
	}
	var number float64
	if err := json.Unmarshal(data, &number); err != nil {
		return fmt.Errorf("fixture float: %w", err)
	}
	*value = Float64(number)
	return nil
}

type Fixture struct {
	SchemaVersion int `json:"schema_version"`
	Upstream      struct {
		Package string `json:"package"`
		Version string `json:"version"`
		Commit  string `json:"commit"`
	} `json:"upstream"`
	Case struct {
		Name        string    `json:"name"`
		Description string    `json:"description"`
		InputKind   string    `json:"input_kind"`
		DType       string    `json:"dtype"`
		Rows        int       `json:"rows"`
		Cols        int       `json:"cols"`
		Data        []Float64 `json:"data"`
		CSR         *struct {
			Data    []Float64 `json:"data"`
			Indices []int     `json:"indices"`
			Indptr  []int     `json:"indptr"`
		} `json:"csr"`
	} `json:"case"`
	Config   map[string]json.RawMessage `json:"config"`
	Expected struct {
		Outcome            string      `json:"outcome"`
		Labels             []int       `json:"labels"`
		Probabilities      []Float64   `json:"probabilities"`
		ClusterPersistence []Float64   `json:"cluster_persistence"`
		OutlierScores      []Float64   `json:"outlier_scores"`
		SingleLinkageTree  [][]Float64 `json:"single_linkage_tree"`
		CondensedTree      []struct {
			Parent, Child int
			Lambda        Float64 `json:"lambda_val"`
			ChildSize     int     `json:"child_size"`
		} `json:"condensed_tree"`
		MinimumSpanningTree [][]Float64 `json:"minimum_spanning_tree"`
		ErrorType           string      `json:"error_type"`
		ErrorMessage        string      `json:"error_message"`
	} `json:"expected"`
	Extended   json.RawMessage `json:"extended"`
	Branches   json.RawMessage `json:"branches"`
	Prediction json.RawMessage `json:"prediction"`
}

func (fixture Fixture) Validate() error {
	if fixture.SchemaVersion != SchemaVersion {
		return fmt.Errorf("schema version %d, want %d", fixture.SchemaVersion, SchemaVersion)
	}
	if fixture.Upstream.Package != "hdbscan" || fixture.Upstream.Version != UpstreamVersion || fixture.Upstream.Commit != UpstreamCommit {
		return fmt.Errorf("invalid upstream identity")
	}
	if fixture.Case.Name == "" || fixture.Case.Rows < 0 || fixture.Case.Cols < 0 {
		return fmt.Errorf("invalid case metadata")
	}
	switch fixture.Case.InputKind {
	case "precomputed_csr":
		if fixture.Case.CSR == nil || len(fixture.Case.CSR.Data) != len(fixture.Case.CSR.Indices) || len(fixture.Case.CSR.Indptr) != fixture.Case.Rows+1 {
			return fmt.Errorf("invalid CSR representation")
		}
	case "dense", "precomputed_dense":
		if len(fixture.Case.Data) != fixture.Case.Rows*fixture.Case.Cols {
			return fmt.Errorf("data length %d does not match shape %dx%d", len(fixture.Case.Data), fixture.Case.Rows, fixture.Case.Cols)
		}
	default:
		return fmt.Errorf("unknown input kind %q", fixture.Case.InputKind)
	}
	switch fixture.Expected.Outcome {
	case "success":
		if len(fixture.Expected.Labels) != fixture.Case.Rows || len(fixture.Expected.Probabilities) != fixture.Case.Rows || len(fixture.Expected.OutlierScores) != fixture.Case.Rows {
			return fmt.Errorf("point output lengths do not match %d input rows", fixture.Case.Rows)
		}
		for _, probability := range fixture.Expected.Probabilities {
			p := float64(probability)
			if math.IsNaN(p) || p < 0 || p > 1 {
				return fmt.Errorf("membership probability %v outside [0,1]", p)
			}
		}
		for _, persistence := range fixture.Expected.ClusterPersistence {
			p := float64(persistence)
			if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 {
				return fmt.Errorf("invalid cluster persistence %v", p)
			}
		}
		if len(fixture.Expected.SingleLinkageTree) != fixture.Case.Rows-1 {
			return fmt.Errorf("single linkage rows=%d want %d", len(fixture.Expected.SingleLinkageTree), fixture.Case.Rows-1)
		}
		for i, row := range fixture.Expected.SingleLinkageTree {
			if len(row) != 4 {
				return fmt.Errorf("single linkage row %d has %d fields", i, len(row))
			}
		}
		if len(fixture.Expected.CondensedTree) == 0 {
			return fmt.Errorf("success fixture has empty condensed tree")
		}
		if fixture.Case.InputKind == "dense" {
			finiteRows := 0
			for row := 0; row < fixture.Case.Rows; row++ {
				finite := true
				for _, value := range fixture.Case.Data[row*fixture.Case.Cols : (row+1)*fixture.Case.Cols] {
					finite = finite && !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
				}
				if finite {
					finiteRows++
				}
			}
			if len(fixture.Expected.MinimumSpanningTree) != finiteRows-1 {
				return fmt.Errorf("minimum spanning tree rows=%d want %d", len(fixture.Expected.MinimumSpanningTree), finiteRows-1)
			}
		}
		for i, row := range fixture.Expected.MinimumSpanningTree {
			if len(row) != 3 {
				return fmt.Errorf("minimum spanning tree row %d has %d fields", i, len(row))
			}
		}
	case "error":
		if fixture.Expected.ErrorType == "" {
			return fmt.Errorf("error outcome has no error_type")
		}
	default:
		return fmt.Errorf("unknown outcome %q", fixture.Expected.Outcome)
	}
	return nil
}

func Load(fileSystem fs.FS, name string) (Fixture, error) {
	var fixture Fixture
	data, err := fs.ReadFile(fileSystem, name)
	if err != nil {
		return fixture, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		return fixture, fmt.Errorf("decode %s: %w", name, err)
	}
	if err := fixture.Validate(); err != nil {
		return fixture, fmt.Errorf("validate %s: %w", name, err)
	}
	return fixture, nil
}

func LoadAll(fileSystem fs.FS) ([]Fixture, error) {
	entries, err := fs.ReadDir(fileSystem, ".")
	if err != nil {
		return nil, err
	}
	fixtures := make([]Fixture, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") || entry.Name() == "SHA256SUMS.json" {
			continue
		}
		fixture, err := Load(fileSystem, path.Clean(entry.Name()))
		if err != nil {
			return nil, err
		}
		fixtures = append(fixtures, fixture)
	}
	return fixtures, nil
}

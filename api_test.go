package hdbscan

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"reflect"
	"sort"
	"testing"
)

// TestPublicAPISnapshot makes exported surface changes explicit in review. The
// compiler and examples separately lock signatures and methods.
func TestPublicAPISnapshot(t *testing.T) {
	want := []string{
		"Algorithm", "ApproximateBackend", "ClusterSelectionMethod", "CondensedEdge",
		"Config", "Dense64", "EOM", "ErrInvalidConfig", "ErrInvalidPrecomputed",
		"ErrInvalidShape", "ErrNonFinite", "ErrPredictionData", "ErrPredictionUnsupported", "ErrTooFewPoints", "Exact", "ExportCSV",
		"ExportFormat", "ExportJSON", "ExtractionConfig", "Fit", "FitPredict",
		"FlatResult", "Leaf", "Linkage", "MSTEdge", "Manhattan", "Metadata",
		"Metric", "MetricFunc", "MinkowskiMetric", "Model", "New", "NewMinkowski",
		"Precomputed", "Reference", "ReferencePrecomputed", "Result", "SelectAlgorithm",
		"SquaredEuclidean", "SquaredMetric", "Euclidean", "AlgorithmApproximate",
		"AlgorithmAuto", "AlgorithmBruteForce", "AlgorithmKDTree", "AlgorithmReference",
		"BallTreeIndex", "BrayCurtis", "Canberra", "Chebyshev", "CutSingleLinkage",
		"ErrDisconnected", "ErrInsufficientNeighbors", "ErrInvalidSparse", "KDTreeIndex",
		"ReferenceSparse", "RobustSingleLinkage", "RobustSingleLinkageConfig", "RobustSingleLinkageResult",
		"SparsePrecomputed", "SpatialIndex", "ValidSpatialIndexes", "ValidityIndex",
	}
	fset := token.NewFileSet()
	packages, err := parser.ParseDir(fset, ".", func(info os.FileInfo) bool {
		return len(info.Name()) < 8 || info.Name()[len(info.Name())-8:] != "_test.go"
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, file := range packages["hdbscan"].Files {
		for _, declaration := range file.Decls {
			switch node := declaration.(type) {
			case *ast.FuncDecl:
				if node.Recv == nil && ast.IsExported(node.Name.Name) {
					seen[node.Name.Name] = true
				}
			case *ast.GenDecl:
				for _, specification := range node.Specs {
					switch spec := specification.(type) {
					case *ast.TypeSpec:
						if ast.IsExported(spec.Name.Name) {
							seen[spec.Name.Name] = true
						}
					case *ast.ValueSpec:
						for _, name := range spec.Names {
							if ast.IsExported(name.Name) {
								seen[name.Name] = true
							}
						}
					}
				}
			}
		}
	}
	got := make([]string, 0, len(seen))
	for name := range seen {
		got = append(got, name)
	}
	sort.Strings(got)
	sort.Strings(want)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("public API changed\n got: %v\nwant: %v", got, want)
	}
}

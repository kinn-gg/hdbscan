package hdbscan

import "context"

// Model is an immutable HDBSCAN configuration. It is safe for concurrent use;
// each Fit returns an independently owned Result.
type Model struct{ config Config }

// New validates configuration values that do not depend on the input shape.
func New(config Config) (*Model, error) {
	normalized, err := normalizeConfig(config, 0)
	if err != nil {
		return nil, err
	}
	return &Model{config: normalized}, nil
}

// Fit runs the configured estimator and honors ctx cancellation.
func (model *Model) Fit(ctx context.Context, x Dense64) (Result, error) {
	if model == nil {
		return Result{}, ErrInvalidConfig
	}
	return Exact(ctx, x, model.config)
}

// FitPredict fits x and returns an owned copy of its flat cluster labels.
func (model *Model) FitPredict(ctx context.Context, x Dense64) ([]int, error) {
	result, err := model.Fit(ctx, x)
	if err != nil {
		return nil, err
	}
	return append([]int(nil), result.Labels...), nil
}

// Fit is a convenience form of New followed by Model.Fit.
func Fit(ctx context.Context, x Dense64, config Config) (Result, error) {
	model, err := New(config)
	if err != nil {
		return Result{}, err
	}
	return model.Fit(ctx, x)
}

// FitPredict is a convenience form of New followed by Model.FitPredict.
func FitPredict(ctx context.Context, x Dense64, config Config) ([]int, error) {
	model, err := New(config)
	if err != nil {
		return nil, err
	}
	return model.FitPredict(ctx, x)
}

// ExtractionConfig controls flat cluster extraction from a retained hierarchy.
type ExtractionConfig struct {
	ClusterSelectionMethod      ClusterSelectionMethod
	AllowSingleCluster          bool
	MaxClusterSize              int
	ClusterSelectionEpsilon     float64
	ClusterSelectionPersistence float64
}

// FlatResult owns the slices produced by hierarchy extraction.
type FlatResult struct {
	Labels             []int
	Probabilities      []float64
	ClusterPersistence []float64
}

// Extract selects flat clusters from the retained hierarchy without recomputing
// distances, neighbors, or the minimum spanning tree.
func (result Result) Extract(options ExtractionConfig) (FlatResult, error) {
	config := result.config
	config.ClusterSelectionMethod = options.ClusterSelectionMethod
	config.AllowSingleCluster = options.AllowSingleCluster
	config.MaxClusterSize = options.MaxClusterSize
	config.ClusterSelectionEpsilon = options.ClusterSelectionEpsilon
	config.ClusterSelectionPersistence = options.ClusterSelectionPersistence
	if config.ClusterSelectionMethod == "" {
		config.ClusterSelectionMethod = EOM
	}
	config, err := normalizeConfig(config, len(result.Labels))
	if err != nil {
		return FlatResult{}, err
	}
	labels, probabilities, persistence := selectClusters(len(result.Labels), result.CondensedTree, stabilities(result.CondensedTree), config)
	return FlatResult{labels, probabilities, persistence}, nil
}

# FLASC branch detection

Branch detection is an opt-in post-processing step over a fitted dense-vector
model. Set `Config.BranchDetectionData` before fitting, then call
`Result.DetectBranches`. Set `PredictionData` as well when new observations must
be assigned to branches.

```go
fitted, err := hdbscan.Fit(ctx, x, hdbscan.Config{
    MinClusterSize: 5,
    BranchDetectionData: true,
    PredictionData: true,
})
branches, err := fitted.DetectBranches(ctx, hdbscan.BranchConfig{
    Method: hdbscan.BranchFull,
})
labels, strengths, clusterLabels, clusterStrengths, branchLabels,
    branchStrengths, err := branches.ApproximatePredict(ctx, novel)
```

`BranchFull` constructs every within-cluster mutual-reachability edge up to the
cluster MST threshold. Its worst-case time and retained output are O(n²).
`BranchCore` extends the cluster MST with the retained k-nearest-neighbor graph;
it is smaller and faster but more sensitive to noise. Precomputed dense and CSR
inputs do not retain feature-space data and return `ErrBranchUnsupported`.

`BranchGraph` stores each undirected edge once. Its two uint32 endpoints are
packed into a uint64 and its centrality/reachability weights live in parallel
float64 slices. `Edge` unpacks one record; `Write` streams CSV or NDJSON without
creating an adjacency representation. Inputs above 2³²-1 rows are outside this
representation and are already impractical for in-memory full FLASC.

The output includes combined labels/probabilities, original cluster values,
branch membership, persistence, centrality, cluster point indices, and one
linkage/condensed tree per cluster. Caller-provided contiguous cluster labels may
override the fitted partition. All output slices are owned by the result.

The pinned upstream fixture uses `hdbscan 0.8.44` and a deterministic three-flare
example. Full-graph partitions exceed 0.96 pairwise agreement under the distinct
stable tie policies. Equal-centrality
MSTs are not unique: Go orders by centrality, reachability, and packed endpoints;
SciPy may choose a different valid tree. Core graphs also intentionally collapse
upstream duplicate endpoint pairs, so incidental raw edge counts are not a parity
requirement.

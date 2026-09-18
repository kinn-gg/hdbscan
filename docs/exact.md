# Memory-efficient exact Euclidean engine

`Exact` is the normal dense Euclidean path for low- and moderate-dimensional
data. It builds a k-d tree over an index permutation and keeps the caller's
contiguous `Dense64.Data` as the only point storage. Tree nodes contain bounds,
not copied observations.

```go
result, err := hdbscan.Exact(ctx, data, hdbscan.Config{
    MinClusterSize: 10,
    MinSamples:     10,
    Workers:        runtime.GOMAXPROCS(0),
})
```

Core-distance searches use a bounded max-heap of `MinSamples` entries per
worker. Workers are fixed for the duration of a search, honor context
cancellation, and write disjoint output entries. Candidate merging and all tie
decisions follow input-index order, so worker counts do not affect results.

The mutual-reachability implementation includes exact, tree-pruned Boruvka with
stable component merging. Because different equal-weight MSTs can produce a
different sequential linkage representation, the exported MST uses a streamed
stable-Prim frontier matching `Reference`'s canonical tie policy. It allocates
`O(n)` frontier storage and never materializes pairwise distances.

For non-Euclidean metrics, zero-dimensional observations, or more than 32
dimensions, Auto uses the bounded-memory blocked brute-force path. The dimension
cutoff is a conservative guard against ineffective k-d-tree pruning. See
[algorithm selection and approximate mode](algorithms.md) for forced policies
and result metadata.

The auxiliary-memory bound is `O(n*d + n + workers*MinSamples)`: tree bounds and
indices are linear in the input size, core-search heaps are bounded, component
scratch is reused by round, and the final MST is preallocated to `n-1` edges.
MST edges use the public compact `{int, int, float64}` struct. On 64-bit targets
its payload is the same 24 bytes per edge as three packed arrays, while the
single slice avoids separate allocations and converts directly to the result.

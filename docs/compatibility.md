# Upstream compatibility contract

## Pinned oracle

The compatibility oracle is
[`scikit-learn-contrib/hdbscan`](https://github.com/scikit-learn-contrib/hdbscan)
release `0.8.44`, tag `release-0.8.44`, commit
`dfdc9ca2b265ab6d50cf64b428199785700af240`. Its BSD-3-Clause notice is retained
in this repository's `LICENSE` and `NOTICE` files.

Generated fixture metadata includes both the package version and commit. Changing
the oracle requires an explicit fixture schema or fixture revision and a review
of every changed result.

## Initial parameter contract

The v0.1 API preserves these upstream semantics for dense and precomputed paths.

| Parameter | Upstream default | Contract |
|---|---:|---|
| `min_cluster_size` | `5` | Integer greater than one; smallest group retained as a cluster. |
| `min_samples` | `None` | Uses `min_cluster_size` when unset; defines the core-distance neighbor. |
| `alpha` | `1.0` | Positive robust-single-linkage scaling used by mutual reachability. |
| `metric` | `minkowski` | Dense v0.1 supports Euclidean, Manhattan, Minkowski and precomputed distances. |
| `p` | `2` | Minkowski power. Ignored by metrics that do not use it. |
| `algorithm` | `best` | Go `Auto` will select an implementation; fixtures force `generic` for a stable oracle. |
| `approx_min_span_tree` | `true` | Exact fixture generation forces `false`; approximation must be explicit in Go result metadata. |
| `cluster_selection_method` | `eom` | `eom` selects stable clusters; `leaf` selects hierarchy leaves. |
| `cluster_selection_epsilon` | `0.0` | Nonnegative distance threshold used while extracting flat clusters. |
| `cluster_selection_persistence` | `0.0` | Nonnegative minimum persistence used during selection. |
| `max_cluster_size` | `0` | Zero means unlimited; affects EOM but not leaf selection. |
| `allow_single_cluster` | `false` | When false, the root is not returned as the sole cluster. |
| `match_reference_implementation` | `false` | Upstream compatibility switch; not planned as a permanent Go option. |
| `cluster_selection_epsilon_max` | `+Inf` | Upper epsilon limit for selection; prediction is not guaranteed to honor non-default values upstream. |

`min_samples` counts the point itself in the pinned upstream implementation's
public semantics. Go tests must lock any boundary behavior before exposing a
stable API.

## Output contract

- Noise labels are `-1`. Non-noise cluster numbers carry no identity across
  implementations and are compared using partitions, not integer equality.
- Membership probabilities and persistence values are finite values in `[0, 1]`,
  compared with absolute tolerance `1e-7` and relative tolerance `1e-6` unless a
  fixture documents a stricter exact expectation.
- GLOSH outlier scores use the same tolerances. Upstream NaN values, when they are
  the defined result for a degenerate case, are encoded as strings in JSON.
- Condensed tree, linkage tree, and MST comparisons canonicalize endpoint order and
  sort equal-weight edges before comparing.

## Partition comparison

Canonical labels are useful for exact fixtures: scan points in input order and
rename the first non-noise cluster to `0`, the next unseen cluster to `1`, and so
on; retain noise as `-1`.

For approximate paths and comparisons where ties admit multiple valid trees, use
pairwise co-membership or adjusted Rand index (ARI). Exact paths target ARI `1.0`.
Approximate quality gates will be dataset-specific and must always publish the
exact-path baseline.

## Determinism and ties

The Go implementation will order an undirected edge as
`(weight, min(endpoint), max(endpoint))`, comparing endpoint indices only after
equal floating-point weights. Parallel candidate sets must be merged using that
order. A fixed input and configuration must yield the same result for all worker
counts.

Upstream outputs for tie-heavy data remain fixtures even where their incidental
cluster numbering is not normative.

## Non-finite values

- Input decoding preserves IEEE `NaN`, `+Inf`, and `-Inf` through explicit JSON
  strings because JSON numbers cannot encode them.
- Dense feature rows containing a non-finite value follow the pinned upstream
  behavior recorded by fixtures. The intended Go API will reject non-finite rows
  by default, with any compatibility filtering behavior made explicit.
- A precomputed dense distance matrix may contain `+Inf` to represent missing
  edges, subject to enough finite neighbors to compute core distances. NaN and
  negative distances are invalid.
- Arithmetic must not silently turn invalid input into a plausible clustering.

## Comparison exclusions

Python-specific estimator behavior, pandas/NetworkX conversion, plotting, joblib
caching, and scikit-learn metadata are outside the Go compatibility contract.
Tree and graph content remains in scope through native Go records and streaming
encoders.

## Python-to-Go feature matrix

| Python `hdbscan` feature | Go v0.1 | Notes |
|---|---|---|
| Dense float64 fitting | Yes | `Model.Fit`, `Fit`; finite row-major input. |
| `fit_predict` | Yes | `Model.FitPredict`, `FitPredict`. |
| Euclidean, Manhattan, Minkowski | Yes | Auto optimization is Euclidean; others use bounded brute force. |
| Chebyshev, Canberra, Bray-Curtis | Yes | Allocation-free metrics using bounded brute force; valid index geometry is declared by `ValidSpatialIndexes`. |
| Dense precomputed distances | Yes | `ReferencePrecomputed`; `+Inf` missing edges supported. |
| EOM and leaf selection | Yes | Also available through retained `Result.Extract`. |
| Selection epsilon/persistence | Yes | Configurable at fit or retained-hierarchy extraction. |
| Probabilities/persistence/GLOSH | Yes | Native owned slices. |
| Condensed/linkage/MST data | Yes | Native records and streaming CSV/NDJSON. |
| Approximate MST | Yes | Explicit opt-in and marked in `Metadata`. |
| New-point prediction | Yes | Opt in with `Config.PredictionData`; approximate labels, strengths, and GLOSH scores. |
| Soft membership vectors | Yes | Single, batched, fitted-point, and caller-buffer APIs. |
| Sparse precomputed graphs | Yes | Symmetric CSR stays sparse; missing entries are absent edges. Disconnected graphs return `ErrDisconnected` with a component count. |
| Robust single linkage/tree cuts | Yes | Shared neighbor, mutual-reachability MST, and linkage representation. |
| DBCV validity index | Yes | Pair work is streamed; no all-pairs matrix is allocated. |
| FLASC branch detection | Yes | Opt-in dense feature state; full/core packed graphs, hierarchies, persistence, membership, and approximate prediction. |
| pandas, NetworkX, plotting, sklearn metadata | No | Python integration surface is intentionally not ported. |

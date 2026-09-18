# Exact reference implementation

`Reference` is the small-data correctness oracle for this package. It computes
all pairwise distances, core distances, the mutual-reachability graph, a dense
Prim minimum spanning tree, the single-linkage and condensed hierarchies, and
then flat clusters with either EOM or leaf selection. Results also contain
membership probabilities, persistence, and GLOSH outlier scores.

The implementation takes `O(n²)` time and memory. `ReferencePrecomputed` copies
the supplied `n × n` matrix before processing it, so callers retain ownership.
Both entry points honor context cancellation and return output slices owned by
the result.

Equal-weight choices are deterministic: Prim scans point indices in ascending
order, retains the first equal candidate, and linkage processing preserves the
resulting order. An undirected edge can be canonicalized as `(distance,
min(from,to), max(from,to))` for comparisons.

Dense feature input must be finite. A precomputed matrix may use positive
infinity for unavailable distances, but the resulting mutual-reachability graph
must remain connected.

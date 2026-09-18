# Distance numerical policy

Dense observations must contain only finite values. Precomputed distance
matrices additionally require a square, symmetric, nonnegative matrix with a
zero diagonal; positive infinity is permitted to represent a disconnected
pair.

Euclidean distance uses a scaled sum of squares. This preserves finite norms
when directly squaring a large component would overflow, and avoids losing a
representable norm solely because its square underflows. Squared Euclidean is
used as the ordering key where an algorithm does not require actual distance
units. As IEEE-754 requires, squared distance can itself overflow even when the
corresponding Euclidean distance is representable.

Scalar accumulation is deterministic for a fixed input and Go architecture.
Future vector kernels must produce bitwise-identical ordering keys or document a
relative tolerance and retain a stable index-based tie breaker. Architecture
dispatch is selected once by the caller, never within a vector loop.

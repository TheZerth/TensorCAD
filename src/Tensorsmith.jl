"""
    Tensorsmith

A computational environment for tensor algebras and their quotients.

## Architecture

The mathematical foundation is the free tensor algebra T(V) of a vector space V.
Every algebra of interest is a quotient T(V)/I, realized as a normalization rule
on multi-indices, so that the product in every algebra is:

    a ·_A b  =  normalize_A( concatenate(a, b) )

## Current phase: 0–1 (scaffold + free tensor algebra)

| Phase | Content                                 | Status      |
|-------|-----------------------------------------|-------------|
| 0     | Scaffold, scalar ring abstraction       | ✓ complete  |
| 1     | Free tensor algebra T(V)                | ✓ complete  |
| 2     | Symmetric and Exterior quotients        | pending     |
| 3     | Metric + Clifford                       | pending     |
| 4     | Inter-algebra maps                      | pending     |
| 5     | Compiled numeric mode (build_function)  | pending     |
| 6     | Visualization                           | pending     |

## Scalar ring

Everything is parametric over a scalar ring `R`.  The default exact ring is
`ExactRing = Rational{BigInt}` for Phases 0–4.  `Float64` works as a fast
ring but does not contain ℚ and cannot be used for section maps.

See `src/scalar_ring.jl` for the full rationale and interface contract.
"""
module Tensorsmith

include("scalar_ring.jl")
include("vector_space.jl")
include("free_tensor.jl")
include("quotient_algebras.jl")

end # module Tensorsmith

"""
    TensorsmithSymbolicsExt

Extension module: loaded automatically when both Tensorsmith and Symbolics.jl
are active in the same session.

    using Tensorsmith, Symbolics

Provides:
  - `contains_rationals` dispatch for `Symbolics.Num` (enables antisymmetrize /
    symmetrize over symbolic coefficients)
  - `symbolic_vars(name, n)` -- create n named symbolic scalars
  - `symbolic_element(space, AlgebraType, name)` -- generic grade-1 AlgebraTensor
  - `symbolic_clifford_vector(metric, name)` -- generic grade-1 CliffordTensor

All existing Tensorsmith operations (+, *, wedge, grade, ...) work unmodified
with R = Symbolics.Num because the core code is fully generic over R.
"""
module TensorsmithSymbolicsExt

using Tensorsmith
using Symbolics

# ─────────────────────────────────────────────────────────────────────────────
# Ring trait

# Symbolics.Num represents symbolic expressions that can carry exact rational
# constants (e.g. 1//6 is representable), so antisymmetrize and symmetrize work.
Tensorsmith.contains_rationals(::Type{Symbolics.Num}) = true

# ─────────────────────────────────────────────────────────────────────────────
# symbolic_vars

"""
    symbolic_vars(name, n) -> Vector{Symbolics.Num}

Return `[name_1, name_2, ..., name_n]` as symbolic scalars.
"""
function Tensorsmith.symbolic_vars(name::Symbol, n::Int)
    [Symbolics.variable(name, i) for i in 1:n]
end

# ─────────────────────────────────────────────────────────────────────────────
# symbolic_element -- generic grade-1 AlgebraTensor

"""
    symbolic_element(space, AlgebraType, name) -> AlgebraTensor{A, Symbolics.Num}

Build the generic grade-1 element `name_1*e_1 + ... + name_n*e_n` in algebra A.
"""
function Tensorsmith.symbolic_element(
    space     :: VectorSpace,
    ::Type{A},
    name      :: Symbol,
) where A
    R = Symbolics.Num
    n = space.n
    terms = Dict{Vector{Int}, R}()
    for i in 1:n
        terms[[i]] = Symbolics.variable(name, i)
    end
    AlgebraTensor{A, R}(space, terms)
end

# ─────────────────────────────────────────────────────────────────────────────
# symbolic_clifford_vector -- generic grade-1 CliffordTensor

"""
    symbolic_clifford_vector(metric, name) -> CliffordTensor{Symbolics.Num}

Build `name_1*e_1 + ... + name_n*e_n` in `Cl(V, g)` over `Symbolics.Num`.

Note: `metric` must itself be a `Metric{Symbolics.Num}`.  To convert an exact
metric, use `symbolic_metric(g)`.
"""
function Tensorsmith.symbolic_clifford_vector(
    metric :: Metric{Symbolics.Num},
    name   :: Symbol,
)
    R = Symbolics.Num
    n = metric.space.n
    terms = Dict{Vector{Int}, R}()
    for i in 1:n
        terms[[i]] = Symbolics.variable(name, i)
    end
    CliffordTensor{R}(metric, terms)
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: lift an exact Metric{Rational{BigInt}} to Metric{Symbolics.Num}

"""
    symbolic_metric(g::Metric{Rational{BigInt}}) -> Metric{Symbolics.Num}

Convert an exact metric to one whose entries are `Symbolics.Num` so it can be
used with symbolic Clifford elements.
"""
function symbolic_metric(g::Metric{Rational{BigInt}})
    R = Symbolics.Num
    Metric{R}(g.space, R.(g.g))
end

export symbolic_metric

end # module TensorsmithSymbolicsExt

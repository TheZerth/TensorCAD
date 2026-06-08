# ── Phase 5: Symbolic ring interface stubs ────────────────────────────────────
#
# These function stubs declare the public API for Symbolics.jl integration.
# The actual methods are defined in ext/TensorsmithSymbolicsExt.jl, which is
# loaded automatically when the user does:
#
#   using Tensorsmith, Symbolics
#
# Calling any of these without Symbolics loaded will raise a MethodError
# with a clear "no method matching" message.
#
# Design principle: the core Tensorsmith code (Phases 0-4) is fully generic
# over any scalar ring R.  Symbolics.Num slots in as R with no changes to
# the algebra code itself.  This file only adds convenience wrappers that
# make the common "create a symbolic element" pattern ergonomic.

"""
    symbolic_vars(name::Symbol, n::Int) -> Vector{Symbolics.Num}

Create n symbolic scalar variables named `name_1, name_2, ..., name_n`.

Requires Symbolics.jl:
    using Tensorsmith, Symbolics
    xs = symbolic_vars(:x, 3)   # [x_1, x_2, x_3]

# Example
    V = VectorSpace(3)
    xs = symbolic_vars(:x, 3)
    # Build a generic grade-1 FreeTensor with symbolic coefficients:
    t = sum(xs[i] * basis_vector(V, Symbolics.Num, i) for i in 1:3)
"""
function symbolic_vars end

"""
    symbolic_element(space, AlgebraType, name::Symbol) -> AlgebraTensor{A, Symbolics.Num}

Create a generic grade-1 element of `AlgebraType` over `space` with symbolic
coefficients `name_1*e_1 + name_2*e_2 + ... + name_n*e_n`.

Requires Symbolics.jl.

# Example
    V = VectorSpace(3)
    a = symbolic_element(V, ExteriorAlgebra, :a)
    b = symbolic_element(V, ExteriorAlgebra, :b)
    prod = a \\wedge b   # symbolic exterior product
"""
function symbolic_element end

"""
    symbolic_clifford_vector(metric::Metric, name::Symbol) -> CliffordTensor{Symbolics.Num}

Create a generic grade-1 Clifford element with symbolic coefficients
`name_1*e_1 + ... + name_n*e_n` in `Cl(V, g)`.

Requires Symbolics.jl.

# Example
    gE = signature_metric(VectorSpace(3), Rational{BigInt}, 3, 0, 0)
    a  = symbolic_clifford_vector(gE_sym, :a)   # over Symbolics.Num metric
    b  = symbolic_clifford_vector(gE_sym, :b)
    prod = a * b   # symbolic geometric product in Cl(3,0)
"""
function symbolic_clifford_vector end

"""
    symbolic_metric(g::Metric{Rational{BigInt}}) -> Metric{Symbolics.Num}

Lift an exact metric to one whose entries are `Symbolics.Num` so it can be used
with symbolic Clifford elements.

Requires Symbolics.jl.
"""
function symbolic_metric end

"""
    isequal_simplified(a, b) -> Bool

Decide equality of two Tensorsmith elements whose scalar ring is
`Symbolics.Num`, canonicalising every coefficient with `Symbolics.expand`
before comparing.  Structural `==` is insufficient for symbolic coefficients
because `x + y` and `y + x` are syntactically distinct even though equal.

Requires Symbolics.jl.  For the exact ring `Rational{BigInt}`, ordinary `==`
is already correct — use that instead.
"""
function isequal_simplified end

export symbolic_vars, symbolic_element, symbolic_clifford_vector,
       symbolic_metric, isequal_simplified

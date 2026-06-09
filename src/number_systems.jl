# ── Phase 7: Number systems + dual-number AD ─────────────────────────────────
#
# DESIGN.md §6: the classical number systems are not additions — they are
# `Cl(p,q,r)` for the right signature, which validates the emergent philosophy:
#
#   ℂ              = Cl(0,1)    (generator squares to −1)
#   split-complex  = Cl(1,0)    (generator squares to +1)
#   dual numbers   = Cl(0,0,1)  (null generator, ε² = 0)
#   quaternions ℍ  = even subalgebra Cl⁺(3,0)   (a sub-algebra, not a scalar ring)
#
# Two roles, both supported here:
#   (1) emergent sub-algebras inside a Clifford algebra — the builders below;
#   (2) the scalar ring R itself — see `Dual{T}` (forward-mode autodiff).
#
# A scalar ring must be commutative, so ℝ, ℂ, dual, and split-complex numbers
# can serve as R; ℍ cannot.  This file adds no new algebra machinery — every
# builder is a thin wrapper over the existing Clifford constructors.

# ── ℂ — complex numbers as Cl(0,1) ────────────────────────────────────────────

"""
    complex_metric(R = ExactRing) -> Metric{R}

Metric of `Cl(0,1)`: a 1-D space whose generator `i` squares to `−1`.
"""
complex_metric(::Type{R} = ExactRing) where R =
    signature_metric(VectorSpace(1, [:i]), R, 0, 1, 0)

"""
    imaginary_unit(R = ExactRing) -> CliffordTensor{R}

The imaginary unit `i` of `Cl(0,1)`, satisfying `i² = −1`.
"""
imaginary_unit(::Type{R} = ExactRing) where R = clifford_basis_vector(complex_metric(R), 1)

"""
    complex_number(a::R, b::R) -> CliffordTensor{R}

The complex number `a + b·i` realized in `Cl(0,1)`.
"""
function complex_number(a::R, b::R) where R
    g = complex_metric(R)
    clifford_scalar(g, a) + b * clifford_basis_vector(g, 1)
end

"""    complex_real(z) -> R — the real part `Re z` (grade-0 coefficient)."""
complex_real(z::CliffordTensor{R}) where R = get(z.terms, Int[], zero(R))
"""    complex_imag(z) -> R — the imaginary part `Im z` (coefficient of `i`)."""
complex_imag(z::CliffordTensor{R}) where R = get(z.terms, [1], zero(R))
"""    complex_conjugate(z) -> CliffordTensor — `a − b·i` (the grade involution)."""
complex_conjugate(z::CliffordTensor) = grade_involution(z)

# ── Split-complex (hyperbolic) numbers as Cl(1,0) ─────────────────────────────

"""
    split_complex_metric(R = ExactRing) -> Metric{R}

Metric of `Cl(1,0)`: a 1-D space whose generator `j` squares to `+1`.
"""
split_complex_metric(::Type{R} = ExactRing) where R =
    signature_metric(VectorSpace(1, [:j]), R, 1, 0, 0)

"""
    split_complex_number(a::R, b::R) -> CliffordTensor{R}

The split-complex number `a + b·j` (`j² = +1`) realized in `Cl(1,0)`.
"""
function split_complex_number(a::R, b::R) where R
    g = split_complex_metric(R)
    clifford_scalar(g, a) + b * clifford_basis_vector(g, 1)
end

# ── Dual numbers as Cl(0,0,1) (the sub-algebra view) ──────────────────────────

"""
    dual_clifford_metric(R = ExactRing) -> Metric{R}

Metric of `Cl(0,0,1)`: a 1-D space whose generator `ε` is null (`ε² = 0`).
The reason the L4 metric layer supports degenerate signatures.

(For dual numbers as a *scalar ring* — the autodiff use — see [`Dual`](@ref).)
"""
dual_clifford_metric(::Type{R} = ExactRing) where R =
    signature_metric(VectorSpace(1, [:ε]), R, 0, 0, 1)

"""
    dual_clifford_number(a::R, b::R) -> CliffordTensor{R}

The dual number `a + b·ε` (`ε² = 0`) realized in `Cl(0,0,1)`.
"""
function dual_clifford_number(a::R, b::R) where R
    g = dual_clifford_metric(R)
    clifford_scalar(g, a) + b * clifford_basis_vector(g, 1)
end

# ── Quaternions ℍ as the even subalgebra Cl⁺(3,0) ─────────────────────────────

"""
    quaternion_metric(R = ExactRing) -> Metric{R}

Euclidean metric of `Cl(3,0)`; ℍ is its even subalgebra `⟨1, e₁e₂, e₂e₃, e₁e₃⟩`.
"""
quaternion_metric(::Type{R} = ExactRing) where R =
    signature_metric(VectorSpace(3), R, 3, 0, 0)

"""
    quaternion_basis(R = ExactRing) -> NTuple{4, CliffordTensor{R}}

Return `(one, i, j, k)`, the quaternion units as bivectors of `Cl⁺(3,0)`:
`i = e₁e₂`, `j = e₂e₃`, `k = e₁e₃`.  They satisfy
`i² = j² = k² = ijk = −1` and `ij = k, jk = i, ki = j`.
"""
function quaternion_basis(::Type{R} = ExactRing) where R
    g = quaternion_metric(R)
    (clifford_one(g),
     clifford_basis_element(g, [1, 2]),
     clifford_basis_element(g, [2, 3]),
     clifford_basis_element(g, [1, 3]))
end

"""
    quaternion(a::R, b::R, c::R, d::R) -> CliffordTensor{R}

The quaternion `a + b·i + c·j + d·k` in `Cl⁺(3,0)` (see [`quaternion_basis`](@ref)).
"""
function quaternion(a::R, b::R, c::R, d::R) where R
    one_, i, j, k = quaternion_basis(R)
    a * one_ + b * i + c * j + d * k
end

# ── Dual numbers as a SCALAR RING — exact forward-mode autodiff ───────────────
#
# `Dual{T}` carries a value and its first derivative: `a + b·ε`, `ε² = 0`.  It
# satisfies the scalar-ring interface, so it can be used as `R` *anywhere* in
# the library with no other changes.  Evaluating any library expression at
# `Dual(a, 1)` yields `f(a) + f′(a)·ε` (DESIGN.md §6): exact derivatives for free.

"""
    Dual{T} <: Number

A forward-mode autodiff dual number `value + deriv·ε` with `ε² = 0`.  Usable as
the scalar ring `R` throughout Tensorsmith; an expression evaluated at
`Dual(a, one(T))` returns `Dual(f(a), f′(a))`.
"""
struct Dual{T} <: Number
    value :: T
    deriv :: T
end

# Julia auto-generates the inner `Dual{T}(::T,::T)` and outer `Dual(::T,::T)`
# constructors; we add construction-from-integer and the identity convert.
Dual{T}(n::Integer) where T = Dual{T}(T(n), zero(T))
Dual{T}(x::Dual{T}) where T  = x

"""    dual_seed(a::T) -> Dual{T} — the AD seed `a + 1·ε` (derivative 1 at `a`)."""
dual_seed(a::T) where T = Dual{T}(a, one(T))
"""    dual_value(x::Dual) — the value part."""
dual_value(x::Dual) = x.value
"""    dual_deriv(x::Dual) — the derivative (ε) part."""
dual_deriv(x::Dual) = x.deriv

Base.zero(::Type{Dual{T}}) where T = Dual{T}(zero(T), zero(T))
Base.one(::Type{Dual{T}})  where T = Dual{T}(one(T),  zero(T))
Base.zero(x::Dual{T}) where T = zero(Dual{T})
Base.one(x::Dual{T})  where T = one(Dual{T})

Base.:+(x::Dual{T}, y::Dual{T}) where T = Dual{T}(x.value + y.value, x.deriv + y.deriv)
Base.:-(x::Dual{T}, y::Dual{T}) where T = Dual{T}(x.value - y.value, x.deriv - y.deriv)
Base.:-(x::Dual{T})             where T = Dual{T}(-x.value, -x.deriv)
# Product rule
Base.:*(x::Dual{T}, y::Dual{T}) where T =
    Dual{T}(x.value * y.value, x.deriv * y.value + x.value * y.deriv)
# Quotient rule
function Base.:/(x::Dual{T}, y::Dual{T}) where T
    v = x.value / y.value
    Dual{T}(v, (x.deriv * y.value - x.value * y.deriv) / (y.value * y.value))
end
# Chain rule for sqrt (lets autodiff flow through `magnitude` over a sqrt ring)
function Base.sqrt(x::Dual{T}) where T
    s = sqrt(x.value)
    Dual{T}(s, x.deriv / (s + s))
end

Base.iszero(x::Dual)            = iszero(x.value) && iszero(x.deriv)
Base.:(==)(x::Dual{T}, y::Dual{T}) where T = x.value == y.value && x.deriv == y.deriv
Base.isequal(x::Dual{T}, y::Dual{T}) where T =
    isequal(x.value, y.value) && isequal(x.deriv, y.deriv)
Base.hash(x::Dual, h::UInt) = hash(x.deriv, hash(x.value, h))

# A Dual ring is sqrt-capable exactly when its underlying scalar type is.
has_sqrt(::Type{Dual{T}}) where T = has_sqrt(T)

# Promote integer/real literals so `2 * x` and friends work without ceremony.
Base.promote_rule(::Type{Dual{T}}, ::Type{S}) where {T, S<:Real} = Dual{promote_type(T, S)}
Base.convert(::Type{Dual{T}}, n::Real) where T = Dual{T}(convert(T, n), zero(T))
Base.convert(::Type{Dual{T}}, x::Dual{T}) where T = x

function Base.show(io::IO, x::Dual)
    print(io, x.value, " + ", x.deriv, "ε")
end

# ── Exports ───────────────────────────────────────────────────────────────────

export complex_metric, imaginary_unit, complex_number,
       complex_real, complex_imag, complex_conjugate,
       split_complex_metric, split_complex_number,
       dual_clifford_metric, dual_clifford_number,
       quaternion_metric, quaternion_basis, quaternion,
       Dual, dual_seed, dual_value, dual_deriv

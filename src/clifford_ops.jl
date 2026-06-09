# ── Phase 7 (L5): Geometric-algebra operation suite on CliffordTensor ─────────
#
# GOVERNING PRINCIPLE (DESIGN.md §3, §11):
#
#   Every derived product is a *grade projection of the single geometric
#   product*, extended bilinearly over terms — never an independent product.
#
# The geometric product `A * B` already lives on CliffordTensor (clifford.jl).
# Each operation here is `grade-select(geometric_product(A_r, B_s))` summed over
# the homogeneous pieces `A_r`, `B_s`.  This mirrors GASmith's "one operation,
# many shadows" design and keeps the invariant honest: change the geometric
# product and every shadow follows.
#
# All coefficient comparisons use `isequal` / `iszero` (never `==` in a boolean
# context) so the symbolic ring (`R = Symbolics.Num`) stays safe.

# ── Involutions ───────────────────────────────────────────────────────────────
#
# Each is a grade-wise sign map.  For a blade of grade r (= length of its
# multi-index key) the sign is:
#
#   grade_involution    (-1)^r
#   reversion           (-1)^{r(r-1)/2}
#   clifford_conjugate  (-1)^{r(r+1)/2}   ( = grade_involution ∘ reversion )

# Apply a grade-wise sign function f(r) ∈ {+1,-1} to every blade of A.
function _grade_sign_map(A::CliffordTensor{R}, f) where R
    terms = Dict{Vector{Int}, R}()
    for (idx, c) in A.terms
        terms[idx] = isodd(f(length(idx))) ? -c : c   # f returns the exponent
    end
    CliffordTensor{R}(A.metric, terms)
end

"""
    grade_involution(A::CliffordTensor) -> CliffordTensor

The main (grade) involution `Â`: multiply each grade-`r` part by `(-1)^r`.
A ring automorphism (`(AB)^ = Â B̂`); on vectors it is negation.
"""
grade_involution(A::CliffordTensor) = _grade_sign_map(A, r -> r)

"""
    reversion(A::CliffordTensor) -> CliffordTensor
    ~A

The reversion `Ã` (reverse the order of vector factors in every blade):
multiply each grade-`r` part by `(-1)^{r(r-1)/2}`.  A ring anti-automorphism
(`(AB)~ = B̃ Ã`).

Named `reversion` rather than overloading `Base.reverse` (which reverses
collections); the unary operator `~A` is provided as the conventional GA
shorthand for `Ã`.
"""
reversion(A::CliffordTensor) = _grade_sign_map(A, r -> div(r * (r - 1), 2))

Base.:~(A::CliffordTensor) = reversion(A)

"""
    clifford_conjugate(A::CliffordTensor) -> CliffordTensor

The Clifford conjugate `Ā`: multiply each grade-`r` part by `(-1)^{r(r+1)/2}`.
Equal to `grade_involution(reversion(A))` (the composite of the two other
involutions), and the ring anti-automorphism that negates vectors.
"""
clifford_conjugate(A::CliffordTensor) = _grade_sign_map(A, r -> div(r * (r + 1), 2))

# ── Grade-projected products ──────────────────────────────────────────────────
#
# CONTRACTION CONVENTION: we use Dorst's left/right contractions (Dorst, Fontijne
# & Mann, *Geometric Algebra for Computer Science*).  "Inner product" has several
# incompatible definitions in the literature, so we expose the two contractions
# explicitly instead:
#
#   wedge              A ∧ B  = Σ_{r,s} ⟨A_r B_s⟩_{r+s}
#   left_contraction   A ⨼ B  = Σ_{r,s} ⟨A_r B_s⟩_{s-r}   (grade s-r, else 0)
#   right_contraction  A ⨽ B  = Σ_{r,s} ⟨A_r B_s⟩_{r-s}   (grade r-s, else 0)
#   scalar_product     ⟨A B⟩_0 (returned as an R, not a CliffordTensor)

# Bilinear extension: for every homogeneous grade pair (r,s) take the geometric
# product and keep its grade-`target(r,s)` part.  `target` returning a value
# outside 0:n simply contributes nothing (homogeneous_component of an absent
# grade is zero).
function _graded_product(A::CliffordTensor{R}, B::CliffordTensor{R}, target) where R
    _check_compatible(A, B)
    acc = clifford_zero(A.metric)
    for r in grades(A), s in grades(B)
        k = target(r, s)
        k < 0 && continue
        prod = homogeneous_component(A, r) * homogeneous_component(B, s)
        acc  = acc + homogeneous_component(prod, k)
    end
    acc
end

"""
    wedge(A::CliffordTensor, B::CliffordTensor) -> CliffordTensor
    A ∧ B

The outer (exterior) product realized *inside* `Cl(V,g)`: the grade-`(r+s)` part
of the geometric product, extended bilinearly.  Metric-independent, and under
the zero metric it agrees with the exterior algebra's `∧` (a tested invariant).
"""
wedge(A::CliffordTensor, B::CliffordTensor) = _graded_product(A, B, (r, s) -> r + s)

"""
    left_contraction(A::CliffordTensor, B::CliffordTensor) -> CliffordTensor
    A ⨼ B

Dorst's **left** contraction: the grade-`(s-r)` part of `A_r B_s`, extended
bilinearly (zero when `s < r`).  Lowers grade: contracting a grade-`r` blade
into a grade-`s` blade yields grade `s-r`.
"""
left_contraction(A::CliffordTensor, B::CliffordTensor) =
    _graded_product(A, B, (r, s) -> s - r)

"""
    right_contraction(A::CliffordTensor, B::CliffordTensor) -> CliffordTensor
    A ⨽ B

Dorst's **right** contraction: the grade-`(r-s)` part of `A_r B_s`, extended
bilinearly (zero when `r < s`).
"""
right_contraction(A::CliffordTensor, B::CliffordTensor) =
    _graded_product(A, B, (r, s) -> r - s)

"""
    scalar_product(A::CliffordTensor{R}, B::CliffordTensor{R}) -> R

The scalar product `⟨A B⟩₀`: the grade-0 coefficient of the geometric product,
returned as an element of the ring `R` (not a `CliffordTensor`).
"""
function scalar_product(A::CliffordTensor{R}, B::CliffordTensor{R}) where R
    _check_compatible(A, B)
    get((A * B).terms, Int[], zero(R))
end

# Unicode operator aliases (consistent with the codebase's ∧/⊗ usage).
∧(A::CliffordTensor, B::CliffordTensor) = wedge(A, B)
⨼(A::CliffordTensor, B::CliffordTensor) = left_contraction(A, B)
⨽(A::CliffordTensor, B::CliffordTensor) = right_contraction(A, B)

# ── Pseudoscalar, dual / Hodge ────────────────────────────────────────────────

"""
    pseudoscalar(metric::Metric{R}) -> CliffordTensor{R}

The unit pseudoscalar `I = e₁∧⋯∧eₙ` (the top-grade canonical blade) of `Cl(V,g)`.
For `n = 0` this is the scalar `1`.
"""
pseudoscalar(metric::Metric{R}) where R =
    clifford_basis_element(metric, collect(1:metric.space.n))

"""
    dual(A::CliffordTensor) -> CliffordTensor
    dual(A::CliffordTensor, metric::Metric)

The (Hodge) dual `A ↦ A I⁻¹`, where `I` is the unit pseudoscalar.  `I⁻¹` exists
only when the metric is non-degenerate; for a **degenerate** metric (`det g = 0`,
signature `r > 0`) the pseudoscalar is null and an `ArgumentError` is thrown,
naming the degeneracy (DESIGN.md §5, honest mechanisms).

Involutivity: `dual(dual(A)) = ±A`, with the sign fixed by the dimension and
signature through `I⁻²`.

The two-argument form is a convenience that checks `metric == A.metric`.
"""
function dual(A::CliffordTensor{R}) where R
    I  = pseudoscalar(A.metric)
    ss = scalar_square(I)
    iszero(ss) && throw(ArgumentError(
        "the unit pseudoscalar is non-invertible: the metric is degenerate " *
        "(det g = 0, signature r > 0), so the Hodge dual is undefined"))
    Iinv = (one(R) / ss) * reversion(I)
    A * Iinv
end

function dual(A::CliffordTensor{R}, metric::Metric{R}) where R
    metric == A.metric || throw(ArgumentError(
        "dual: supplied metric does not match the element's metric"))
    dual(A)
end

# ── Magnitude / norm ──────────────────────────────────────────────────────────

"""
    scalar_square(A::CliffordTensor{R}) -> R

The scalar square `⟨A Ã⟩₀ = scalar_product(A, reversion(A))`, returned as an `R`.
Exact for every ring (it uses no square root); may be negative or zero for
indefinite signatures.  This is the squared magnitude when it is non-negative.
"""
scalar_square(A::CliffordTensor) = scalar_product(A, reversion(A))

"""
    magnitude(A::CliffordTensor{R}) -> R

The magnitude `√⟨A Ã⟩₀`.  **Requires a `sqrt`-capable ring** (`has_sqrt(R)`):
the exact ring `Rational{BigInt}` is rejected because the square root of an
exact scalar is generally irrational — use [`scalar_square`](@ref) for an exact
value there.  For an indefinite signature `scalar_square(A)` may be negative, in
which case the magnitude is imaginary (use a complex ring).
"""
function magnitude(A::CliffordTensor{R}) where R
    has_sqrt(R) || throw(ArgumentError(
        "magnitude requires a sqrt-capable ring (has_sqrt(R) == true); " *
        "$R is exact and its square root may be irrational. Use " *
        "scalar_square(A) for the exact squared magnitude, or work over " *
        "Float64 / Symbolics.Num."))
    sqrt(scalar_square(A))
end

# ── Versor inverse ────────────────────────────────────────────────────────────

# True iff x is exactly the scalar 1 (its only term is the grade-0 coefficient 1).
_is_scalar_one(x::CliffordTensor{R}) where R =
    length(x.terms) == 1 && haskey(x.terms, Int[]) && isequal(x.terms[Int[]], one(R))

"""
    inv_mv(A::CliffordTensor{R}) -> CliffordTensor{R}

The inverse of a **blade or versor**: `A⁻¹ = Ã / ⟨A Ã⟩₀` when `scalar_square(A)`
is non-zero.  The candidate is verified to actually invert `A` (two-sided),
which rejects null elements and non-versor multivectors.

The **general** multivector inverse is out of scope for this phase; for
non-versor input (or `scalar_square(A) = 0`) an `ArgumentError` is thrown
explaining the limitation.
"""
function inv_mv(A::CliffordTensor{R}) where R
    ss = scalar_square(A)
    iszero(ss) && throw(ArgumentError(
        "scalar_square(A) = 0: A is null and has no inverse"))
    cand = (one(R) / ss) * reversion(A)
    (_is_scalar_one(A * cand) && _is_scalar_one(cand * A)) || throw(ArgumentError(
        "A is not a versor; the general multivector inverse is out of scope " *
        "for this phase. Only blades/versors (A⁻¹ = Ã/⟨AÃ⟩₀) are supported."))
    cand
end

# ── Rotors ────────────────────────────────────────────────────────────────────

"""
    apply_rotor(rotor::CliffordTensor, x::CliffordTensor) -> CliffordTensor

Apply a rotor by the sandwich product `R x R̃` (reversion of `R` on the right).
For a unit rotor this is an isometry — it preserves [`scalar_square`](@ref) —
and is the workhorse for rotations and boosts.  `x` may be any multivector.
"""
function apply_rotor(rotor::CliffordTensor{R}, x::CliffordTensor{R}) where R
    rotor * x * reversion(rotor)
end

"""
    rotor_exp(B::CliffordTensor{R}) -> CliffordTensor{R}

Closed-form exponential of a **bivector** `B`, whose square `B²` is a scalar `s`:

  - `s < 0` (`s = -θ²`):  `cos θ + (B/θ) sin θ`   — a rotation rotor;
  - `s > 0` (`s =  φ²`):  `cosh φ + (B/φ) sinh φ` — a boost rotor;
  - `s = 0`:              `1 + B`.

**Requires a transcendental, real-ordered ring** (`has_transcendentals(R)`),
because it takes `√|s|` and a sin/cos (or sinh/cosh) and must compare `s` to `0`.
`Float64` works; the exact ring `Rational{BigInt}` cannot represent a general
rotor — use [`rotor_exp_series`](@ref) for an exact, truncated alternative.

Throws `ArgumentError` if `B` is not a bivector or `B²` is not a scalar.
"""
function rotor_exp(B::CliffordTensor{R}) where R
    has_transcendentals(R) || throw(ArgumentError(
        "rotor_exp needs sqrt/sin/cos/sinh/cosh and a sign decision on B², so " *
        "it requires a transcendental real-ordered ring (has_transcendentals(R)). " *
        "$R does not qualify; use rotor_exp_series(B, order) for an exact, " *
        "in-ring truncation."))
    isempty(B.terms) && return clifford_one(B.metric)          # exp(0) = 1
    grades(B) == [2] || throw(ArgumentError(
        "rotor_exp expects a pure bivector (grade 2); got grades $(grades(B))"))
    B2 = B * B
    (isempty(B2.terms) || grades(B2) == [0]) || throw(ArgumentError(
        "rotor_exp requires B² to be a scalar; got grades $(grades(B2))"))
    s = get(B2.terms, Int[], zero(R))
    one_mv = clifford_one(B.metric)
    if iszero(s)
        one_mv + B
    elseif s < zero(R)
        θ = sqrt(-s)
        cos(θ) * one_mv + (sin(θ) / θ) * B
    else
        φ = sqrt(s)
        cosh(φ) * one_mv + (sinh(φ) / φ) * B
    end
end

"""
    rotor_exp_series(B::CliffordTensor{R}, order::Int) -> CliffordTensor{R}

Truncated exponential `Σ_{k=0}^{order} Bᵏ / k!`, valid for **any** `B` and
**staying in the ring** — the exact-ring counterpart of [`rotor_exp`](@ref).
Requires `R ⊇ ℚ` (`contains_rationals(R)`) for the `1/k!` factors; over
`Rational{BigInt}` it is exact.  For a bivector with `B² = -θ²` it converges to
the true rotor `cos θ + (B/θ) sin θ` as `order → ∞`.
"""
function rotor_exp_series(B::CliffordTensor{R}, order::Int) where R
    contains_rationals(R) || throw(ArgumentError(
        "rotor_exp_series divides by k!; it requires R ⊇ ℚ (contains_rationals(R))."))
    order >= 0 || throw(ArgumentError("order must be ≥ 0, got $order"))
    term = clifford_one(B.metric)      # k = 0 term: B⁰/0! = 1
    acc  = term
    for k in 1:order
        term = (one(R) / R(k)) * (term * B)   # term_k = term_{k-1} * B / k = Bᵏ/k!
        acc  = acc + term
    end
    acc
end

# ── Exports ───────────────────────────────────────────────────────────────────

export grade_involution, reversion, clifford_conjugate,
       wedge, left_contraction, right_contraction, scalar_product,
       ∧, ⨼, ⨽,
       pseudoscalar, dual,
       scalar_square, magnitude,
       inv_mv,
       apply_rotor, rotor_exp, rotor_exp_series

# QRCS Experiment 2, Part B — Weighted-Lattice Dispersion Relation

Engine commit: `c1b9ef0` (L10.1 WeightedGridBase). Derivation pass; zero source files changed.

## Dispersion relation (derived; full work in derive.jl header)

For a uniform lattice with constant grade weights `(w₁, w₂)` (edge, face),
the wave operator `K_w = δ_w d = W₁⁻¹ dᵀ W₂ d` on the grade-1 potential `A`
has, per Bloch wavevector `k = (k_x, k_y)`, a longitudinal (gauge) mode at
eigenvalue 0 and a single transverse mode at

    λ(k; w₁, w₂) = (w₂/w₁) · [ (2 − 2cos k_x) + (2 − 2cos k_y) ].

The weights enter ONLY through the scalar ratio `w₂/w₁`. With `ω² = λ`,
the long-wavelength (`|k|→0`) wave speed and refractive index are

    c_medium(w₁,w₂) = √(w₂/w₁),    n(w₁,w₂) = c_vac/c_medium = √(w₁/w₂).

Direction: larger edge weight `w₁` ⇒ larger `n` ⇒ slower wave; larger face
weight `w₂` ⇒ faster. Dielectric dictionary (from the L10.1 energy split):
`ε = w₁`, `μ = 1/w₂`, so `n = √(εμ) = √(w₁/w₂)` and `Z = √(μ/ε) = 1/√(w₁ w₂)`.
**Index depends on the ratio `w₁/w₂`; impedance depends on the product `w₁ w₂`.**

## Prediction 1 — uniform SCALAR weight produces NO refraction

Exact over `Rational{BigInt}`: the grade-1 wave operator `K_w` for a uniform
scalar weight `w = 3` on all grades vs unit weight `w = 1`.

| grid | K_w(3,3,3) == K_w(1,1,1) exactly | nonzero spectrum (identical) |
|------|----------------------------------|------------------------------|
| (2,2) | true | 2.0, 4.0, 4.0, 6.0 |
| (3,2) | true | 1.5858, 3.0, 3.5858, 4.4142, 5.0, 6.4142 |

**CONFIRMED.** `K_w(3,3,3) == K_w(1,1,1)` as exact rational matrices (not
merely isospectral — literally the same operator), because `K_w = (1/w)dᵀ(w)d
= dᵀd` when `w₁ = w₂ = w`. A uniform scalar weight rescales the energy
(inner product) but leaves the dynamics — hence the wave speed — untouched,
so it does not refract. (The grade-0 weight `w₀` never enters `K_w` at all.)

## Prediction 2 — asymmetric grade weights DO refract

Exact: `(w₁,w₂) = (2,1)` vs unit, on GridBase(2,2). Closed form predicts the
whole spectrum scales by `w₂/w₁ = 1/2`.

    K_w(2,1) == (1/2)·K_unit exactly :  true
    unit nonzero spectrum  : 2.0, 4.0, 4.0, 6.0
    (2,1) nonzero spectrum : 1.0, 2.0, 2.0, 3.0
    ratios (should all be 1/2): 0.5, 0.5, 0.5, 0.5

**CONFIRMED.** Asymmetric `(w₁,w₂)` changes the nonzero spectrum (here exactly
halves it ⇒ wave speed × `√(1/2)`), so it refracts — as required for Part A
to be well-posed.

## Dispersion verification — the closed form (5) is exact on finite grids

The face sine-mode `u_{p,q}(i,j) = sin(πp(i+1)/(N+1))·sin(πq(j+1)/(M+1))` is
an exact eigenmode of `M₂^w = d δ_w` (which shares the nonzero `K_w` spectrum
by the SVD pairing). Below: applying the engine operator returns a CONSTANT
ratio (spread ≈ 0) equal to the closed form, on GridBase(2,1) and (3,2).

| grid | (p,q) | engine λ | closed form (w₂/w₁)[Σ 2−2cos] | spread |
|------|-------|----------|-------------------------------|--------|
| (2,1) | (1,1) | 3.0 | 3.0 | 0.0 |
| (2,1) | (2,1) | 5.0 | 5.0 | 0.0 |
| (2,1) | (1,1) | 0.75 | 0.75 | 0.0 |
| (2,1) | (2,1) | 1.25 | 1.25 | 0.0 |
| (3,2) | (1,1) | 0.39644661 | 0.39644661 | 0.0 |
| (3,2) | (2,1) | 0.75 | 0.75 | 0.0 |

**CONFIRMED.** Engine eigenvalues match (5) to <1e-10; spreads are roundoff.

## Long-wavelength index — convergence to `c_medium = √(w₂/w₁)`

Lowest nonzero mode `(p,q)=(1,1)` on `N×N` grids, `(w₁,w₂)=(4,1)` ⇒ target
`c_medium = √(1/4) = 1/2`, `n = √(4/1) = 2`. Wavenumber `|k|² = 2(π/(N+1))²`,
`c = √(λ/|k|²)`. Discrete factor `(2−2cosθ)/θ² = 1 − θ²/12 + …` ⇒ 2nd order.

| N  | engine λ(1,1) | measured c_medium | target √(w₂/w₁) | error |
|----|---------------|-------------------|-----------------|-------|
|  4 | 0.19098301 | 0.49181582 | 0.5 | 0.00818418 |
|  8 | 0.06030738 | 0.49746539 | 0.5 | 0.00253461 |
| 16 | 0.0170269 | 0.49928883 | 0.5 | 0.00071117 |
| 32 | 0.00452808 | 0.49981121 | 0.5 | 0.00018879 |

Error-ratio per N-doubling: 3.23, 3.56, 3.77 (→ 4 = second-order as N grows; θ = π/(N+1) slightly inflates small-N ratios).
`c_medium → 0.5` and `n = 1/c_medium → 2`, matching the closed form.

## Index vs impedance — ratio vs product (exact)

Common scaling `(w₁,w₂)=(2,1)` vs `(4,2)` (c=2): `K_w` identical exactly? true.

So `n` is invariant under common scaling — `n` depends on the RATIO `w₁/w₂`
(here 2 in both ⇒ `n=√2`), confirming the operational scaling test. The
impedance `Z = 1/√(w₁ w₂)` does change (product 2 → 8, `Z` ×1/2): impedance
depends on the PRODUCT. Measuring `Z` needs the reflection coefficient at an
interface — deferred to Part A (the packet machinery); here it is established
structurally from the energy split. **NB: this swaps product/ratio relative
to the PM's conjectured form** (which guessed `n ∝ product`); the conjecture's
scaling test is satisfied, its product/ratio attribution is not.

## Part A target (the pre-registered, E1-style closed-form number)

Part A (the refraction experiment) must launch a long-wavelength wave packet
on the potential `A` across an interface between a unit region `(w₁,w₂)=(1,1)`
and a medium region with asymmetric weights, and measure the ratio of phase
(or group) speeds across the interface. The **pre-registered target** is

    c_medium / c_vacuum = √(w₂/w₁),   equivalently   n = √(w₁/w₂),

with Snell's law `sin θ_i / sin θ_t = n_medium / n_vacuum = √(w₁/w₂)` for an
oblique interface. Concretely, the registered medium `(w₁,w₂) = (4,1)` must
give a measured `n = 2.000` (speed halved) to the packet's resolution; a
uniform scalar medium `(w,w)` must give `n = 1` (NO bending) as the null
control. A uniform scalar weight that bends the packet, or an asymmetric
weight whose measured `n` departs from `√(w₁/w₂)` beyond discretization
error, is a finding about the model/operator — to be reported, not tuned.

## Reading discipline (Ledger §5)

This establishes the **kinematic** index of metric-variation-as-medium — the
exact, safe half of §5 (Plebanski re-description). It makes NO claim of
gravity control, EM-tunable `c`, or inexact co-variation; those remain on the
constrained fork. The result is `n = √(w₁/w₂)` (ratio), DERIVED here so Part A
measures against an independently-known target, in the E1 discipline.

---
Self-check summary: scalar-null CONFIRMED; asymmetric-refraction CONFIRMED; dispersion-formula EXACT-MATCH; common-scaling-invariance CONFIRMED.

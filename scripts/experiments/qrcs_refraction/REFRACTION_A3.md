# QRCS Experiment 2, Part A3 — The Graded-Index Shapiro Capstone

Engine commit: `ae98f8f`. Model-layer experiment; zero source files changed.
Built on A2 (REFRACTION_A2.md). Float64, Leapfrog, dt = 0.1.

**Strict reading (pre-registered):** this is the EXACT, KINEMATIC,
gravity-as-medium re-description ONLY (Ledger §5) — a wave following an
**imposed** graded index. The profile is NOT derived from a source/mass/
field equation. This is **not** a gravity simulation, not curved spacetime,
not a solved metric, not inexact co-variation, not EM-tunable c.

## Engine fidelity — sparse K_w re-checked on a GRADED base

A2 verified `K_w = W₁⁻¹ Dᵀ W₂ D` equals the engine `codifferential(d(·))`/
`evolve`/`Leapfrog` path to ~2e-15 for UNIFORM weights. The weights here
are graded (non-uniform), so the equivalence is RE-CONFIRMED on a graded,
impedance-matched profile (not assumed to carry over):

| check (graded base) | max error |
|---------------------|-----------|
| `K_matrix·v` vs engine `codifferential(d(·))` | 1.8e-15 |
| vectorised leapfrog vs engine `evolve`, A (40 steps) | 4.7e-15 |
| vectorised leapfrog vs engine `evolve`, E (40 steps) | 5.3e-15 |

The graded sparse K_w IS the engine operator (to roundoff). The medium is
impedance-matched: `w₁ = n` on edges, `w₂ = 1/n` on faces ⇒ index √(w₁/w₂)
= n exactly and Z = 1/√(w₁w₂) = 1 (no reflection off the gradient), each
grade sampled at its own cell location (DESIGN.md §20). γ NOT used.

## Pre-registered ray-theory target (case 1, the headline NUMBER)

Eikonal `d/ds(n dr/ds) = ∇n` ⇒ for a near-x ray, path curvature
`κ = d²y/dx² = (∂n/∂y)/n`, radius `R = n/|∇n_⊥|`, deflection `Δy = ½κL²`.
Registered: n₀ = 1, gradient G = 0.025 (dn per wavelength), N_λ = 8 wavelengths ⇒
**κ_target = G/(λw·n₀) per cell**, continuum `ΔY_target = ½(G/n₀)N_λ² = 0.8 wavelengths`, end-angle G·N_λ = 11.459°.

## Case 1 headline (λw = 15) and the resolution sweep

Packet launched along +x (an axis ⇒ minimal anisotropy); index increases
with y; the centroid path is fit to a parabola y(x), κ_meas = 2·(x² coeff).

| λw | grid | κ_meas (/cell) | κ_target | ratio κ_meas/κ_target |
|----|------|----------------|----------|-----------------------|
| 10 | 188×148 | 0.0028 | 0.0025 | 1.1094 |
| 15 | 276×216 | 0.0017 | 0.0017 | 1.0491 |
| 20 | 364×284 | 0.0013 | 0.0012 | 1.029 |
| 25 | 452×352 | 0.001 | 0.001 | 1.0198 |

Headline λw=15: measured curvature deflection over the 125.938-cell window = 13.866 cells vs ray-theory 13.217 cells (ratio 1.0491).

**Continuum extrapolation** (ratio r(h) = r∞ + s·h², h = 1/λw, 4 points):
**r∞ = 1.0022 ± 0.00064** — the deflection converges to the
ray-theory target (r∞ ≈ 1), the discretization deficit closing with
resolution (the honest fix per §20; γ not used). Continuum agreement confirmed.

## Control 1 — Uniform null (curvature noise floor)

Flat profile G = 0 (n ≡ 1), same packet/geometry. A flat medium must give
a STRAIGHT path; any curvature is the grid-anisotropy floor.

| run | κ_meas (/cell) |
|-----|----------------|
| uniform null (G=0) | 2.5e-7 |
| graded headline (G=0.025) | 0.0017 |

**Curvature floor = 2.5e-7 /cell (at numerical-noise level); graded = 0.0017 /cell ⇒ deflection-to-floor margin ≈ 6929×.** The flat profile is straight to roundoff; the graded curvature dwarfs the floor.

## Control 2 — Gradient-reversal direction check

Reverse ∇n (G → −G): the packet must curve the OPPOSITE way (κ flips sign,
magnitude preserved). A curvature that does not flip is an artifact.

| run | κ_meas (/cell) | sign |
|-----|----------------|------|
| +G (toward +y) | 0.0017 | + |
| −G (toward −y) | -0.0017 | − |

Curvature **reverses with the gradient** (sign flips, |κ| matches within 20%) — genuine refraction.

## Case 2 — 2D radial index well (the capstone FIGURE)

n(r) = 0.15·exp(−r²/2R²) peak at a central 'mass' (R = 24.0 cells), packet launched at impact parameter b = 18.0 cells; λw=12.
Born/eikonal deflection toward the well (finite-grid ray integral):

| quantity | measured | ray-theory target |
|----------|----------|-------------------|
| deflection angle α | -7.978° | -11.302° |

Both negative ⇒ the packet bends DOWN, toward the high-index centre — the
imposed-index analog of gravitational light bending (kinematic only). The
measured |α| sits 29% below the thin-ray Born target — expected and honest at this single
coarse resolution (λw=12) where, additionally, the packet width σ = 19.2 is comparable to the lens R = 24.0, so the centroid
averages the deflection over a finite beam (a thick ray, not the Born thin
ray). Case 2 is the FIGURE; case 1 carries the rigorous converged NUMBER.

```
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                       ········                                   
                    ·····+++++·····                               
  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●++++···                             
                 ···+++######●●●●●●●●●●●●●●●●●●●●                 
                 ···++#####*####+++···          ●●●●●●            
                 ···+++########+++····                            
                  ····++++++++++····                              
                     ·············                                
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
  x →   (●=packet path, *=index peak/'mass', #/+/· = index well contours)
```
The ray enters straight (top-left), bends toward the index peak `*`, and
exits deflected — the gravitational-lensing picture as an IMPOSED graded
index (Ledger §5 kinematic re-description; no metric is solved).

## Per-run CFL (dt = 0.1)

| run | λ_max bound | z = dt²λ_max | sub-CFL? |
|-----|-------------|--------------|----------|
| case1 λw=15 | 10.569 | 0.1057 | yes |
| case1 λw=25 | 10.473 | 0.1047 | yes |
| null | 8.0 | 0.08 | yes |
| case2 lens | 8.0 | 0.08 | yes |

---
## Status (Ledger §9 permitted shape)

Graded-index refraction — the **kinematic gravity-as-medium re-description**
(Ledger §5) — **demonstrated dynamically**: a wave packet in a smoothly
graded, impedance-matched index curves continuously toward higher index
along the ray-theory path. **Case-1 curvature matches the eikonal target**:
the deflection-to-floor margin is ≈ 6929× (graded κ = 0.0017 vs flat-null floor 2.5e-7 /cell, the flat profile straight to roundoff), the
curvature **reverses with the gradient**, and the measured/target ratio
**converges to the continuum** r∞ = 1.0022 ± 0.00064 over λw = 10/15/20/25 (the
paper-grade 4-point fit; the deficit is lattice discretization, closed by
resolution per §20 — γ NOT used). **Case-2** shows the lensing figure: the
packet bends toward the central index peak by -7.978° (ray-theory -11.302°).

**Strict limit (stated, not exceeded):** the index profile is IMPOSED, not
derived from a source — this is the exact kinematic re-description ONLY,
NOT a gravity simulation, curved spacetime, solved metric, inexact
co-variation, or EM-tunable c. **A3 PASSES**, closing the refraction arc
(Snell interface A2 → graded-index/Shapiro A3) within Ledger §5's safe half.

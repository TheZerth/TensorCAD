# QRCS Experiment 2, Part A2 — The Angled-Interface Snell Bend

Engine commit: `ceb24eb`. Model-layer experiment; zero source files changed.
Built on A1 (REFRACTION_A1.md). Float64, Leapfrog, dt = 0.1.

Method: a transverse Gaussian wave packet (`A₀ = δ_w(env·cos k·x)`,
`E₀ = ω₀·δ_w(env·sin k·x)`, ω₀ = √λ(k) unit-region) is launched obliquely
from a unit region 1 across an axis-aligned vertical interface (normal x̂)
into medium region 2. The packet centroid is the intensity-weighted
position with envelope-intensity proxy `I = a² + (e/ω₀)²`; the propagation
direction is its velocity (least-squares slope vs time). θ is measured from
the interface normal. λw = 10 cells/wavelength, σ = 1.6·λw (headline).

## Engine fidelity (the fast-path licence)

The L10 blackboard stepper costs ~1.7 s/step over ~50k edges — infeasible
for this ~8-run battery. The weighted-base contract states the operator
exactly: `δ_w = W_{g−1}⁻¹ dᵀ W_g`, `K_w = W₁⁻¹ Dᵀ W₂ D` (D = engine face–
edge incidence). This script assembles K_w/δ_w as sparse matrices from the
engine's own boundary signs and verifies equivalence on a (4,1) interface grid:

| check | max error |
|-------|-----------|
| `K_matrix·v` vs engine `codifferential(d(·))` | 8.9e-16 |
| vectorised leapfrog vs engine `evolve`/`Leapfrog`, A (40 steps) | 2.7e-15 |
| vectorised leapfrog vs engine `evolve`/`Leapfrog`, E (40 steps) | 2.2e-15 |

The matrices ARE the engine operator (verified to roundoff); the fast path
is the same Yee/Leapfrog scheme, verified-equivalent. All physics below
runs on this verified operator.

## Pre-registered target

`n = √(w₁/w₂)`; Snell `sin θ₂ = sin θ₁ / n₂`. Unit→(4,1): n₂ = 2.000.
Headline θ₁ = 30° ⇒ sin θ₂ = 0.25 ⇒ **θ₂ = 14.48°** (target, fixed before
measuring). Curve: θ₂(θ₁) = asin(sin θ₁ / 2).

## Headline: unit → (4,1) at θ₁ = 30°

| quantity | measured | target |
|----------|----------|--------|
| θ₁ (incident) | 31.785° | 30° (nominal launch) |
| θ₂ (transmitted) | 17.924° | 14.478° |
| sin θ₁ / sin θ₂ = n | 1.7115 | 2.000 |
| transmitted energy fraction | 0.601 | (Z-mismatch ⇒ <1) |

Grid 222×154 (68752 edges), 1870 steps; θ₁ window 448 samples, θ₂ window 833 samples; windows well-sampled and reflection/boundary-clean (valid).

## Control 1 — Zero-contrast null (headline noise floor)

Identical media both sides (1,1)|(1,1), same packet and θ₁ = 30°. With no
real interface, any bend is pure grid artifact and SETS THE NOISE FLOOR.

| run | θ₁ meas | θ₂ meas | bend |θ₂−θ₁| |
|-----|---------|---------|------------|
| null (1,1)|(1,1) | 31.196° | 31.962° | 0.766° |
| headline (4,1) | 31.785° | 17.924° | 13.861° |

**Null bend = 0.766° ; headline bend = 13.861° ⇒ bend-to-floor margin ≈ 18.101×.** The refractive bend dwarfs the floor.

## Control 2 — Grid-anisotropy quantification

Free packet (no interface) on the unit grid, speed measured along an axis
(θ=0°) vs along the 45° diagonal, at λw = 10 (the headline resolution).
Lattice waves travel slightly faster diagonally — the known confound,
quantified here so the bend's significance is explicit.

| direction | measured speed |
|-----------|----------------|
| axis (0°) | 0.9462 |
| diagonal (45°) | 0.9704 |

Anisotropy (c₄₅−c₀)/c₀ = **2.6%** at λw = 10 (group speed;
waves run faster diagonally, as expected). This directional bias is what
biases the measured angles, and its ANGULAR manifestation is measured
directly by the zero-contrast null (Control 1): the null's 0.766° spurious bend IS the angle-domain artifact. The headline bend (13.861°) exceeds it 18.101×, so the bend is the weight's, not the grid's.

## Control 3 — Multi-angle: one Snell curve, one n

Unit→(4,1) at θ₁ = 20°, 30°, 40°. A genuine index fits ALL angles with one
n; a grid artifact will not.

| θ₁ (nominal) | θ₁ meas | θ₂ meas | θ₂ target | n = sinθ₁/sinθ₂ |
|--------------|---------|---------|-----------|------------------|
| 20° | 21.689° | 12.471° | 9.847° | 1.7115 |
| 30° | 31.785° | 17.924° | 14.478° | 1.7115 |
| 40° | 41.384° | 22.666° | 18.747° | 1.7156 |

Single-n fit (sin θ₁ = n·sin θ₂ through origin): **n = 1.7136** (target 2.000), max residual in sin θ = 0.00076.

## Control 4 — Multi-resolution: convergence of n toward the target

Headline (4,1) at θ₁ = 30° at λw = 10, 15, 20 cells/wavelength (finer
relative to the wavelength). The DISCRIMINATING question (the stop
condition): does the recovered n DRIFT erratically (a spurious artifact),
or CONVERGE toward √(w₁/w₂) = 2.000 as the lattice is refined (a genuine
index whose deficit is the discretization's, vanishing with h)?

| λw | grid | θ₂ meas | n recovered | deficit 2−n |
|----|------|---------|-------------|-------------|
| 10 | 222×154 | 17.924° | 1.7115 | 0.2885 |
| 15 | 327×226 | 15.764° | 1.9112 | 0.0888 |
| 20 | 432×297 | 15.175° | 1.9745 | 0.0255 |

Deficit shrinks monotonically 0.2885 → 0.0888 → 0.0255 (ratios 3.247×, 3.488× per refinement step).
**n CONVERGES toward 2.000** — the deficit closes faster than the 2.25× per step of pure 2nd order (the small transmitted angle amplifies Part B's 2nd-order dispersion error via n = sinθ₁/sinθ₂), so the gap is the lattice's discretization, NOT a spurious drift. PASS.

## Control 5 — Direction: bending away from the normal

Unit→(1,4): n₂ = √(1/4) = 0.5 < 1, so the packet must bend AWAY from the
normal (θ₂ > θ₁), opposite to the (4,1) case. θ₁ = 20° ⇒ target sin θ₂ =
2·sin 20° ⇒ θ₂ ≈ 43.16°.

| case | θ₁ meas | θ₂ meas | n recovered | target n |
|------|---------|---------|-------------|----------|
| unit→(1,4) | 21.251° | 44.758° | 0.5148 | 0.500 |

Bends AWAY from the normal (θ₂ > θ₁) — correct direction.

## sin θ₂ vs sin θ₁  (● measured · `/` target n=2 · `.` fitted n=1.714)

```
                                              
                                             .
                                          .●..
                                      ....  //
                                   ....  //// 
                                ..●. /////    
                             ..../////        
                          ....////            
                       ..●/////               
                   ....////                   
                ...////                       
             ...////                          
          ...///                              
       ...//                                  
    .../                                      
  ../                                         
  sinθ₁ → 0 .. 0.7   (sinθ₂ ↑ 0 .. 0.45)
```
The three angles are collinear through the origin (one index), landing
just ABOVE the n=2 target line and ON the fitted n=1.714 line — the
anisotropy deficit that closes with resolution (Control 4).

## Per-region CFL (worst region sets dt)

| run | λ_max(w) bound | z = dt²λ_max | sub-CFL? |
|-----|----------------|--------------|----------|
| headline (4,1) | 8.0 | 0.08 | yes |
| null (1,1) | 8.0 | 0.08 | yes |
| (1,4) opp | 31.999 | 0.32 | yes |
| λw=15 (4,1) | 8.0 | 0.08 | yes |

dt = 0.1 is sub-CFL (z ≪ 4) in every region; the worst case is the
(1,4) medium (region-2 λ_max scales ×w₂/w₁ = 4).

---
## Status (Ledger §9 permitted shape)

Angled-interface refraction **demonstrated dynamically and attributed to
the weight, not the grid.** An oblique packet crossing a weighted-lattice
interface bends toward the normal in (4,1); the **(4,1) headline bend is 13.861° against a zero-contrast null floor of 0.766° — a bend-to-floor margin of ≈ 18.101×**, far above the
resolved grid anisotropy (2.6% at λw=10). One refractive
index fits ALL incidence angles (20/30/40°) — single-n = 1.7136, max sin-residual 0.00076 — and the (1,4) case bends correctly
AWAY from the normal (n = 0.5148 ≈ 0.5). **Caveat (not
tuned, not floor-subtracted):** the absolute index is anisotropy-limited at
λw=10 — recovered n = 1.7115 vs target √(w₁/w₂) = 2.000 —
and CONVERGES monotonically toward the target with resolution (n = 1.7115 → 1.9112 → 1.9745 at λw = 10/15/20), the expected discretization behaviour (Part B's
2nd-order dispersion error, amplified at small θ₂), confirming the
deficit is the lattice's, not the index law's. No claim of
gravity control, EM-tunable c, or inexact co-variation — kinematic
Plebanski re-description only (Ledger §5). **A2 PASSES on attribution; the
quantitative n is resolution-limited at λw=10 and converging to 2.0.**

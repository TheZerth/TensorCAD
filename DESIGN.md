# TensorCAD / Tensorsmith — Design Charter

**Project:** TensorCAD (engine package: `Tensorsmith`)
**Author:** Kainaan Riordan
**Status:** Living document · Rev 1.1 · 2026-06-08
**Rev 1.1:** added §12 (demonstrable-constants example content for L9/L11).
**Purpose of this file:** the north-star. It records *what we are building, why, and — just as importantly — what we are deliberately not building*. Phase prompts and plugin decisions should reference it. When a choice is unclear, the principles and the build-criterion below decide it.

---

## 1. North Star

TensorCAD is a focused, interactive environment for exploring **algebras of geometric quantities and the calculus of how those quantities change** — and for using that to experiment with new equations, model systems, and visualize their behavior. The proving grounds are electromagnetism and experimental/extended electrodynamics: e.g. expressing Maxwell as a single multivector equation, representing the EM field as one bivector object, and using duality/contraction to view fields in their natural (often lower-effective-dimensional) representation.

The mathematical spine: start from a vector space, build the free tensor algebra `T(V)`, and obtain every other algebra by imposing a multiplication rule or quotient on it — symmetric, exterior, Clifford — with metrics layered on for geometric and Riemannian structure, and the ability to run that **in reverse** (take a Clifford / exterior / symmetric object, lift to `T(V)`, and find its equivalent forms across systems).

## 2. Scope Boundary — what this is *not*

This is the most important section, because the main risk to the project is scope creep dressed as relevance. TensorCAD is **not** a unification of "all mathematics of change," a proof assistant, a category-theory workbench, a control-systems toolbox, a stochastic-processes library, or a general scientific-computing platform. Those are either *consumers* of TensorCAD or *framings* of it. The default answer to "should we add X?" is **no — unless X passes the build-criterion in §4.**

## 3. Design Philosophy

1. **General at the bottom, specialize downward.** `T(V)` (ordered multi-indices, repetition allowed) is the foundation; every named algebra is a quotient of it. A representation that is *specialized* at the bottom (e.g. an antisymmetry-native bitmask) is a leaf, not a trunk — see GASmith, §11.
2. **One product, many normalizers.** An algebra is captured by a `normalize(alg, idx, coef)` rule selected by dispatch; the product in every algebra is `normalize ∘ concatenate`. Adding an algebra is adding a method — nothing else changes.
3. **Multiple dispatch as the core abstraction.** The operation depends on which algebra the operands inhabit; dispatch *is* that. This is the reason the engine is in Julia.
4. **Exact-first; numeric as a compiled view.** Default ring `Rational{BigInt}` for exactness (the reverse-conversion/equivalence feature cannot be decided in floating point). `Symbolics.Num` slots in as another ring `R`. Fast `Float64`/compiled kernels come later via `build_function`: one symbolic definition, two execution modes (the precise mode / realtime mode duality).
5. **Honest mechanisms, surfaced explicitly.** Degeneracy throws rather than silently misbehaving; operations that require ℚ say so; maps document whether they are algebra isomorphisms or only linear ones.
6. **Generic over `R`; zero non-stdlib dependencies in core.** Optional capabilities (e.g. Symbolics) are weak-dependency package extensions.
7. **One element interface.** Every element type subtypes `AbstractTensorElement{R}` and shares grading/equality/projection, so generic code, the UI, and plugins can operate on "an element of *some* algebra" uniformly.
8. **Emergent by default.** Build a thing in core only when it cannot emerge from the foundation (see §4).

## 4. The Build-Criterion (the guardrail)

Implement something in core **only if** it is one of:
- **(a) a mathematical object/operation that does not emerge** from the existing foundation;
- **(b) a numerical or evaluation method that cannot emerge from algebra** (e.g. a mesh discretization, an integrator);
- **(c) a workflow capability**, not a mathematical object (e.g. the equation blackboard, visualization).

Otherwise: **derive it** (it's emergent and costs no new subsystem), or **classify it as a consumer / framing / plugin candidate and do not build it.** "The math has a famous name" is not a reason to build it.

## 5. Architecture — layer map

Layers built and planned. Arrows of dependency run downward; each layer is generic over `R`.

| Layer | Contents | Status |
|---|---|---|
| L0 Scalar ring `R` | exact `Rational{BigInt}` default; `Symbolics.Num` via extension; number systems (ℂ, dual, hyperbolic) as `R`; dual numbers ⇒ autodiff | done + §6 extensions |
| L1 Vector space | `VectorSpace` `V` (equality is dimension-only); dual `V*` | done |
| L2 Free tensor algebra | `T(V)`, concatenation product | done |
| L3 Quotient algebras | `Sym(V)`, `Λ(V)` via normalizers; `Cl(V,g)` via metric | done |
| L3.5 Element interface | `AbstractTensorElement{R}` + hooks | done |
| L4 Metric & tensor calculus | contraction, trace, musical isos (raise/lower), exact inverse metric | done (Phase 6) |
| L5 GA operation suite | involutions, contractions, wedge-in-Cl, dual/Hodge, norm, inverse, rotors (`exp`) | **next** |
| L6 Maps & equivalence engine | projections/sections, symbol maps, cross-system equivalence (symbolic canonicalization) | partial → planned |
| L7 Base space / bundle / field | manifold/grid/graph base; fibre attach; field abstraction | **architectural decision pending** |
| L8 Differential operators | `∇`, `d`, `∇_μ`, Lie derivative — emergent on L7 | planned |
| L9 Equation blackboard | symbolic field variables, equations as objects, variational/Lagrangian derivation | planned |
| L10 Simulation engine | DEC, symplectic integrators, `build_function` compile-to-fast | planned |
| L11 Visualization / UI | multivector/field rendering; interactive exploration | planned |

Cross-cutting: extensibility seams (§8) and the GASmith oracle (§11).

## 6. Scalar Rings and Number Systems

The classical number systems are not additions — they are `Cl(p,q,r)` for the right signature, which validates the emergent philosophy:

- **ℂ** = `Cl(0,1)` (unit bivector squares to −1) — the "imaginary unit" is a geometric bivector.
- **Hyperbolic / split-complex** = `Cl(1,0)` (generator squares to +1).
- **Dual numbers** = `Cl(0,0,1)` (null generator, ε²=0) — the reason L4's metric supports degenerate signatures.
- **Quaternions ℍ** = even subalgebra `Cl⁺(3,0)` — lives as a sub-algebra, *not* as a scalar ring.

Two roles, both supported: as **emergent sub-algebras** inside a Clifford algebra (no new code), and as the **scalar ring `R`** itself. Dividing line: a scalar ring **must be commutative**, so ℝ, ℂ, dual, and hyperbolic numbers can be `R`; ℍ cannot. Consequence worth its weight: **`R` = dual numbers gives exact forward-mode automatic differentiation across the entire library for free** (`f(a+bε) = f(a) + f′(a)·b·ε`); hyper-dual numbers extend this to exact second derivatives.

## 7. The Calculus of Change — taxonomy and the sort

"Change" is many things; the value of cataloguing them is to decide where each lives. Applying §4:

**Emergent — built as ONE thing, grade-projected, never as a zoo:**
- Automatic differentiation = a scalar-ring choice (dual `R`), not a subsystem.
- The field derivative `∇`: `grad`, `div`, `curl`, `Laplacian`, exterior derivative `d`, Dirac operator `D` (`D²=Δ`), monogenic functions (`∇f=0`, Clifford analysis) are all shadows of one operator (`∇F = ∇·F + ∇∧F`).
- Rotors, bivector generators, and geometry Lie groups (rotations, boosts, conformal) = exponentials of bivectors.
- Covariant derivative, connection, curvature, torsion, nonmetricity, holonomy, Berry phase, Lie derivative = the L7/L8 differential-geometry layer; curvature is the commutator of covariant derivatives, holonomy is accumulated loop transport.

**Deliberate tools — pass the build-criterion (evaluation methods or workflow capabilities):**
- **Discrete Exterior Calculus (DEC)** — the simulation method that preserves `d²=0` discretely; the natural home for grid/graph field simulation.
- **Variational / Lagrangian layer** — write an action, take a symbolic functional variation, get Euler–Lagrange field equations. The concrete mechanism for "experiment with new equations" / extended electrodynamics.
- **Symplectic / structure-preserving integrators** — DEC's partner on the dynamics side.
- **Transforms (Laplace/s-domain, Fourier)** — an *optional* analysis layer, explicitly scoped to **linear** subproblems. Nonlinear models are differentiated and simulated, not transformed.
- **Persistent homology** — a *later* analysis tool on simulated field data; serves "recognize and explore patterns." Does not emerge from algebra.
- Plus the equation blackboard (L9) and simulation engine (L10) as workflow capabilities.

**Outside — consumers (use TensorCAD; not in it):** control/cybernetics, information theory, stochastic processes/SDEs, most thermodynamics, graph/network dynamics, Kalman filtering. (Information geometry may *ride on* L8 later as a downstream application — a Fisher metric on a probability manifold — but is not foundation.)

**Outside — design inspiration only (not implemented):** the comonad is the exact shape of a local-neighborhood field update (informs the DEC/stencil design); inter-algebra maps are functorial in spirit; sheaf "local-to-global gluing" is realized concretely by the **bundle** abstraction (L7), not by topos theory.

**Outside — framing / philosophy (not features):** the theory-of-computation block (lambda calculus, Y combinator, domain theory, recursion theory), the category/logic/type-theory frameworks, and the Spencer-Brown / process-philosophy / autopoiesis material. Note: the "recursion" you care about (iterated update rules, flows) is *dynamical-systems iteration*, served by L10 simulation + persistent-homology analysis — **not** computability theory. **Renormalization group** is a long-horizon research *application* of the scale story, not a foundational operator.

## 8. Extensibility (plugin discipline)

**Emergent-from-foundations and plugin-extensibility are the same discipline viewed twice:** both are served by stable seams. Julia makes most of this nearly free — a third-party package can add methods to our public functions for its own types, at full performance, with no registration. So "plugin support" is mostly *naming and freezing the seams we already have* and promising (via semver) not to break them.

**Public extension points (documented as intended surfaces, governed by semver):**
- the `normalize` seam — add a new algebra as a tag type + method;
- the scalar-ring interface — add a new `R` (number systems, AD rings, etc.);
- `AbstractTensorElement{R}` + its hooks — add a new element type;
- (once built) the L7 **bundle interface** — the seam most future plugins will attach to.

**Cost tiers:**
- *Cheap (dispatch-based):* new algebras, rings, element types, operations. The "system" is documentation + semver.
- *Moderate (registry-based):* differential operators, simulation backends, UI panels — the host must *discover* them, so this needs `AbstractOperator` / `AbstractSolver` / `AbstractPanel` interfaces + a registry. This is the part to actually design, later.
- *Hard (plugin-independent):* e.g. a category-theory workbench — hard because of its own math/engineering, made *optional and external* by plugins, not made easy.

**Discipline now (near-free), formal framework later:** document the intended extension surfaces; design the L7 bundle layer as an *interface* (abstract types + a small method contract) rather than one concrete implementation; adopt semver so public-vs-internal is real. Defer the formal plugin framework (registry, abstract operator/solver/panel types, plugin template repo) until after the bundle layer stabilizes. The items deliberately cut in §7 become the natural first candidates for community plugins.

## 9. Roadmap

**Done:** Phases 0–6 — scalar ring; vector space; `T(V)`; `Sym`/`Λ` quotients; `Cl(V,g)`; inter-algebra maps; Symbolics extension; tensor-calculus core (dual space, mixed-variance tensors, contraction/trace, musical isos) + the `AbstractTensorElement` refactor, plus the metric `isequal` and `inverse_metric` fixes.

**Next — finish the algebraic backend:**
1. **GA operation suite** (L5): involutions, contractions, wedge-in-Cl, dual/Hodge, norm, inverse, rotors. *(see companion Claude Code prompt)*
2. Number-systems module + dual-number AD verification (§6) — cheap, proves foundation completeness.
3. Linear maps / outermorphisms.
4. GASmith oracle bridge (fixture cross-check).

**Then:** cross-system **equivalence engine** (L6) — depends on real symbolic canonicalization (`expand`/`simplify`); the headline feature and the hardest.

**Then (architectural):** **base-space / bundle / field** (L7) → **differential operators** (L8, emergent).

**Then:** **equation blackboard / variational layer** (L9) and **simulation engine** (L10, incl. DEC + integrators + compile), developed together (shared Symbolics backbone).

**Then:** **visualization / UI** (L11). First milestone demo: reproduce `∇F = J`, visualize a plane EM wave, then try the dimensional-reduction representation.

*Parallelizable:* the equivalence engine and deeper differential geometry can proceed alongside other work.

## 10. Open Architectural Decisions

1. **Base-space / bundle abstraction** — the next real architectural choice (how coordinates/grids/graphs are specified; how the fibre attaches). Design as an interface from the outset, since fields, DEC, connections, and most plugins hang off it. Note: QRCS-style models are literally a Clifford bundle over a graph, so the bundle abstraction must cover manifold, grid, *and* graph bases.
2. **UI host** — Julia-native (Pluto/Makie) vs. a web front end driving a Julia kernel. Still open; affects the visualization-plugin surface most.
3. **How far into differential geometry before UI** — a GA/tensor *explorer* could ship on the algebraic core first, with connections/curvature added behind it later.
4. **Plugin framework timing** — formalize after L7 stabilizes (§8).

## 11. GASmith's Role

GASmith (C++) is a correct, performant Clifford engine whose core blade is an antisymmetry-native bitmask — i.e. *specialized at the bottom*, a leaf, not a foundation. Its role: **numeric oracle** for the Clifford leaf (Tensorsmith's Clifford output must match it blade-for-blade, via the JSON fixture format), an optional **fast numeric backend**, and a source of **proven design patterns** — notably "all products are grade projections of one product," which L5 follows directly.

## 12. Candidate Demonstrations — "constants as operations" (L9 / L11 example content)

A recurring idea worth capturing *as scoped example content, not as machinery*: a constant is best characterized by **the operation that makes it inevitable**, not by its digits. This is exactly the geometric-algebra view the engine already embodies, and the genuinely valuable, *testable* version is to **demonstrate** these characterizations in the blackboard (L9) and visualization (L11) layers rather than assert them in prose. Candidate vignettes, each exercising capability the engine already has or is slated to have:

- **`i` = the unit bivector quarter-turn.** In our Clifford layer the imaginary unit literally *is* the unit bivector — the rotation generator in a plane. Demo: show `e_{12}² = −1` and that `e_{12}` rotates a vector 90°.
- **`e` = the invariant of self-proportional change.** The eigen-relation `d/dx eˣ = eˣ`; demonstrable once L8's derivative and the dual-number AD ring (Phase 7 / §6) exist — evaluate and *see* the fixed-rate-of-change property.
- **`π` = the invariant of rotational closure.** The rotor angle that returns a vector to itself; demonstrable with the Phase-7 `rotor_exp` / `apply_rotor` suite — sweep the angle and watch closure.
- **`φ` = a fixed point of recursive proportion.** The fixed point of `x ↦ 1 + 1/x`; demonstrable with the L10 iterated-map / fixed-point capability.

**Boundary (do not let this re-inflate):** the broader thesis these came wrapped in — "number arises from distinction; transcendental constants are transport laws; bits are one projection of richer informational primitives" — is a **framing**, filed under §7 "not built." It is interpretive, not an object the engine implements, and the build-criterion (§4) excludes it. Three caveats keep the framing honest if it ever resurfaces: (1) transcendence is *field-relative* (π is algebraic over ℚ(π)), so it is not an intrinsic ontological property; (2) **mathematical** constants (`e, π, φ, i` — dimensionless, necessary) must never be silently grouped with **physical** ones (`c, ℏ, G` — dimensionful, contingent, unit-artifact values); (3) Shannon's bit already *is* the unit of distinguishability, so geometric structure is a richer *carrier*, not evidence that "information isn't bits." Only the operationally-defined, demonstrable characterizations above enter the system — as L9/L11 example content, adding no new machinery.

---

*End Rev 1.1. Amend deliberately; this file is the tie-breaker.*

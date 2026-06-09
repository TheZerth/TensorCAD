# TensorCAD / Tensorsmith — Design Charter

**Project:** TensorCAD (engine package: `Tensorsmith`)
**Author:** Kainaan Riordan
**Status:** Living document · Rev 1.6 · 2026-06-08
**Rev 1.1:** added §12 (demonstrable-constants example content for L9/L11).
**Rev 1.2:** added §13 (the `BaseSpace` contract — four obligations + metric as an optional derived capability); resolved Open Decision 1 from §10.
**Rev 1.3:** added §14 (Potentials are primary, fields are derived) and §15 (the differential-operator arc L8/L8.1/L8.2 + the open L8.1 curvature-representation question).
**Rev 1.4:** settled the L8.1 transport architecture in §15 — two first-class bundle transports (two-sided geometric + one-sided gauge), not unified; Q from inter-node metric variation, R/T from edge holonomy.
**Rev 1.5:** added §15.2 (L8.1 output representation — holonomy-of-a-loop primary, R/T/Q as Fields, lossless primitive with derived views).
**Rev 1.6:** added §16 (geometric objects the visualization layer will want to expose — the dual complex and a deliberate sweep of others; recorded as L11 capabilities, deferred to L11).
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

1. **Base-space / bundle abstraction** — **RESOLVED (see §13).** The base is an *interface* with four obligations plus an optional metric capability; manifold, DEC grid, and Clifford-bundle-over-graph are swappable realizations. What remains is the three-case pressure test and the implementation prompt.
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

## 13. The `BaseSpace` Contract (L7)

The base space is an **interface**, not a commitment to a fundamental ontology. A smooth charted manifold, a DEC cell complex / grid, and a Clifford-bundle-over-graph (the QRCS case) are all *equal, swappable realizations* of one contract. Committing the foundation to any single one (cells-first or manifold-first) would repeat the GASmith error of specializing at the bottom (§3, §11) and would stop the tool from being a neutral instrument for testing models that assume the *other* — so neither is primary. Fields live *over* the base; fibres (Clifford or tensor elements) live *in* the fibre attached at each cell.

**Four obligations** (the minimum `∇`, `d`, and field transport are written against):

1. **Cell enumeration by grade.** The base exposes its cells indexed by dimension (0-cells/nodes, 1-cells/edges, 2-cells/faces, … up to top). A *k-field* assigns fibre elements to k-cells. A graph may stop at 1-cells; a grid/manifold provides the full complex.
2. **Signed incidence / boundary.** For each k-cell, the (k−1)-cells that bound it, **with orientation carried as ±1 signs**. This operator *is* the discrete exterior derivative `d` (the coboundary is the transpose of incidence), so `d` — and grad/curl/div as its grade shadows — come for free on any base that implements it. **Orientation is folded in here, not a separate obligation:** orientation only ever acts through these signs; a non-orientable complex is handled by a base honestly reporting that consistent signs are unavailable, not by a parallel orientation system.
3. **Fibre attachment.** A rule assigning to each cell the fibre algebra it carries (uniform `Cl(3)` for QRCS; tangent/cotangent spaces for a manifold). The fibre is one of the existing element types — this connects L7 to L0–L5.
4. **Transport along a 1-cell.** A map carrying a fibre element across an edge between endpoint frames (QRCS's `τ_uv`; a manifold's parallel transport; a discrete connection). This is what `∇` is written against. Composing transport around a closed loop is what makes the **three failure modes measurable**: rotational holonomy = curvature `R`, translational closure failure = torsion `T`, magnitude drift = nonmetricity `Q`. These three are the independent pieces of a general affine connection, available on *any* base implementing transport.

**One optional capability — the metric.** The metric is **not** a peer obligation; it is an optional, *derived*, *local* capability. A base declares `has_metric` (parallel to the existing `has_sqrt` / `has_transcendentals` traits, §6/Phase 7) and, if true, provides a method returning the **bilinear form per cell/region** plus a **declared signature**. Rationale, with three independent votes: NCG isolates the Dirac operator `𝒟`, DEC isolates `d` from the Hodge star `⋆`, SDG keeps smooth structure logical rather than metric — and QRCS's own account treats the metric as a coarse-grained quantity *derived from* transport and edge weights, not a primitive. Consequences: metric-free operators (`d`, incidence, the natural pairing/contraction) work on every base; metric-dependent operators (`⋆`, `δ`, raising/lowering, magnitude) require `has_metric`. The metric is **local** (per cell/region), so a metric *gradient* between regions is representable — the one genuinely structural nugget from the QRCS domain-wall material (a boundary between metric regions *is* nonmetricity `∇g ≠ 0`). **Signature is base-declarable** (the Lorentzian `(+,−,−,−)` of spacetime is as valid as Euclidean), reusing the existing `Cl(p,q,r)` signature support.

**Deliberately excluded from the contract** (so the one interface serves all three cases): global coordinates (manifold-only; a graph has no embedding) and any primitive global distance (always emergent from weights/transport, never supplied). A *causal poset* base (via the Alexandrov-topology specialization preorder) is a plausible **fourth realization** for later, behind manifold/grid/graph.

**Scope boundary (per §4).** This section defines the *engine* interface only. The QRCS physical apparatus that motivated parts of it — Weyl-compensated scaling, counter-space/nonmetricity activation, conditional metric "awakening," domain-wall refraction, and the `Cl(3)` dimensional-selection argument — is **Layer-2 physical modeling that runs *on* TensorCAD**, not engine structure. It is framing/hypothesis (§7), with the domain-wall-as-nonmetricity scenario logged as candidate L9/L10 example content. *(Note: the 3D-selection bandwidth argument is suggestive but unproven — the vector/bivector counts scale linearly/quadratically, not "exponentially," and equality-as-stability-condition is asserted, not derived; it stays out of the engine regardless.)*

## 14. Potentials Are Primary; Fields Are Derived

**Principle.** The connection (the *potential*) is a first-class, storable object; field strengths (curvature, and EM field strength) are *derived* from it by holonomy/differentiation — **never the reverse.** The engine must not treat field strength `F` as primary with the potential as a convenient fiction.

**Why.** This is the Aharonov–Bohm-correct, Feynman-preferred stance: AB demonstrates the connection `A` has physical effect (an electron phase shift = the holonomy `∮A·dl` = enclosed flux) in a region where the field strength is identically zero, so the potential carries information the field does not. It is also what the user's experimental electrodynamics requires (potentials as the real objects; fields as a description of their interaction/exchange).

**This is already the L7 architecture, made explicit.** The load-bearing base obligation is `transport` (the connection, `τ_uv`); curvature/torsion/nonmetricity are *computed* by composing transport around loops (§13, obligation 4). In gauge terms the connection is the potential and the holonomy is the field strength. A discrete substrate has no `F` until you take the holonomy of the connection — so the substrate cannot be field-first even if one tried. §14 records this as an intended commitment rather than an accident of L7.

**What this forbids baking in (sandbox neutrality, per §2).** The engine must remain agnostic about — never hardcode — flat metric, metric-compatibility, torsion-freeness, the validity of Green's reciprocity, or field-primacy. The metric-affine generality the user's models need is exactly what L7 already provides (transport is an arbitrary connection; the three failure modes are independently representable; the metric is optional/local). Specific physical claims (e.g. reciprocity breakdown between conductors at very different drift velocities; the metric-affine unification of gravity and EM) are **hypotheses to test in the sandbox, not premises to design into it.**

**A language/physics distinction to keep honest (§4 discipline).** The geometric-algebra form of Maxwell (`∇F = J`) is *mathematically equivalent* to the Heaviside vector-calculus form on standard spacetime — it is a better *language* (unifies the four equations, makes potential and bivector structure manifest), not different *physics*. Genuine physical departures come from added structure (the metric-affine extension), not from "geometric product vs. curl/div." Track which results come from notation and which from new physics.

## 15. The Differential-Operator Arc (L8 / L8.1 / L8.2)

L8 is split into three phases by capability dependency (mapping onto the §13 gates), so topological bugs are never conflated with metric/connection bugs:

- **L8 (Tier 1) — topological, metric-free.** The exterior derivative `d` on cochains/fields and its grade shadows grad/curl/div. Needs only `boundary` (obligation 2); works on *every* base including the bare graph; exact. Definition-of-done: `d² = 0` as a strict operator identity, and discrete Stokes `∫_∂Ω ω = ∫_Ω dω` on a mesh. **`d` and the geometric/vector derivative `∇` are honestly distinct operators on a discrete complex** — `d` takes a boundary, `∇` takes a transport — documented to converge in the continuum limit but never unified in the API (that would be a leaky abstraction hiding discrete reality). The charter's "shadows of one operator" (§7) is a *continuum* statement; it discretizes into two mechanisms.
- **L8.1 (Tier 2) — geometric, needs `transport`.** The covariant derivative `∇`, and extraction of curvature/torsion/nonmetricity. Per §14, the connection (potential) is the primary stored object. **Two failure modes come from transport, one does not** (see §15.1): **curvature `R` and torsion `T` are derived from loop holonomy** of the edge-connection (a quantity evaluated on 2-cells / faces), while **nonmetricity `Q` is *not* carried by transport at all** — it is the inter-node variation of the local metric (§13 metric capability), measured along an edge. Versor-conjugation transport is metric-preserving by construction, so it *cannot* carry `Q`; forcing it to would require a shearing (non-orthogonal) map that breaks the Clifford relation `v² = η(v,v)` across the edge. Edges rotate (R/T); nodes deform (Q).
- **L8.2 (Tier 3) — needs `can_hodge` (both metric + dual complex).** Hodge star `⋆`, codifferential `δ`, Hodge–Laplacian `Δ = dδ + δd`. Gated by design — does not exist on the bare graph. This tier makes Maxwell-as-`∇F = J` and the plane-EM-wave demo expressible; that demo is the **final boss of the differential arc** (definition-of-done for L8.2), not L8.

### 15.1 L8.1 transport architecture — SETTLED (Rev 1.4)

The L8.1 opening question — how the EM/gauge potential relates to the geometric connection — is **resolved in favor of two distinct, first-class transports, not a unification.** The decision and its reasoning:

**Two transport kinds, on distinct bundles over the same base:**
- **Geometric / frame transport — two-sided (conjugation).** The spin connection is a *bivector*-valued generator; transport is `M ↦ τ M τ⁻¹` with `τ = exp(B)` a versor (this is L7's `VersorTransport`). It must be two-sided because it has to preserve grade and the quadratic form. Its loop holonomy yields curvature `R` and torsion `T`.
- **Gauge transport — one-sided (representation action).** A gauge connection (e.g. U(1)) acts on a charged section by `ψ ↦ exp(iθ) ψ`. This is a connection on a *principal* bundle, a structurally different object: U(1) is an abelian circle group, not a rotation plane, so its one-sided action is correct, not a wart to be unified away.

**Why not unify them into a single Clifford fibre (the rejected option).** Putting the EM potential `A` in as a grade-1 object in the *same* algebra and forcing it through conjugation does not deliver real unification: a grade-1 vector does not naturally generate a phase by conjugation, so every route to make it (pseudoscalar `exp(Iθ)`, Kaluza–Klein lift, or a bespoke non-conjugation action) either reintroduces a one-sided action under a Clifford costume, smuggles in an unobservable dimension (KK — rejected per §2), or builds a bespoke mechanism we'd own entirely. The "single arena" would be nominal; the two objects would still act by different rules. Bivector-rotation and U(1)-phase are *different groups*; honest geometry keeps them distinct.

**Why this does not betray potential-primacy (§14).** A gauge connection one-form on a principal bundle is itself a *primary, physical* object — Aharonov–Bohm is literally its holonomy. Potential-primacy is satisfied by **either** architecture, so it does not force unification. The real, narrower choice is only whether U(1) phase is represented honestly as a circle (separate bundle) or awkwardly embedded as a not-quite-rotation in the geometric algebra.

**Sandbox-neutrality is the deciding factor (§2).** Hardcoding "EM is a grade-1 sector of the unified Clifford fibre" would bake the user's extended-ED *hypothesis* into the substrate. The neutral engine supports **both** transport kinds as composable mechanisms; whether EM is "really" a separate U(1) or "really" a grade-1 sector of the geometric algebra then becomes a **model built and compared on top of the engine**, not a decision frozen into it. The unified-fibre EM remains a legitimate research construction at the model layer — and the open research question there is what *non-conjugation* action a grade-1 potential has on a state and whether it is consistent — but it is not the L8.1 interface.

**L8.1 interface consequence:** transport is general at the contract level (an invertible fibre map, L7 obligation 4); L8.1 ships *two* realizations — the two-sided `VersorTransport` (already present) and a one-sided gauge/representation transport — and `∇` is written to accept either. Even-subalgebra restriction is a per-realization optimization, never an interface constraint.

### 15.2 L8.1 output representation — SETTLED (Rev 1.5)

How the covariant derivative and the three failure modes are *returned*. Decided so L9 (equations), L10 (simulation), and L11 (visualization) consume a stable, principled shape. Throughline: **every output is a `Field`** over the appropriate grade of cell, so all outputs compose with the L8 operators and arithmetic and render through whatever L11 builds for fields. And the *primitive* stored objects are potential-like (connection, holonomy); the geometric tensors are derived views (§14, one level up).

- **Connection** — the primary stored object: a `Field` over 1-cells (edges) valued in transport maps (the potential, §14). R/T derive from it; it is never derived from them.

- **Holonomy — primitive, lossless, a function of a loop (not a field).** Holonomy is a **function of an oriented, based cycle** (an ordered list of oriented edges) returning the composed transport map. It is **defined on every base, including the bare graph** (which has loops but no faces) — this is the deciding reason it is loop-valued rather than 2-cell-valued: making it a 2-cell field primary would leave curvature *undefined on the QRCS graph substrate*, specializing at the bottom (the recurring §3 error). Subtleties to honor, not paper over: transport composition is **non-abelian**, so the cycle is ordered and oriented (reverse orientation → inverse holonomy); holonomy is basepoint-independent only **up to conjugacy** (different basepoint → conjugate value `gτg⁻¹`), so the **gauge-invariant content is the conjugacy class / its trace** (`Tr(holonomy)`), which is the quantity to expose as the honest invariant.

- **Holonomy field over 2-cells — derived convenience**, available only when `top_grade(b) ≥ 2`: evaluate the loop-function on each face's boundary loop under a fixed orientation/basepoint convention (from the incidence signs + a canonical based vertex). Field *values* are basepoint-convention-dependent; the **trace field is the invariant**. On a bare graph this derived view simply does not exist (no faces), and that is correct.

- **Curvature `R` (bivector) and torsion `T` (vector) — derived FUNCTIONS, never stored.** `curvature`/`torsion` are *pure functions* over the holonomy (the log-extraction of the loop holonomy), returning `Field`s valued in Clifford bivectors (R) / vectors (T). They are **computed on demand, not cached** — caching a derived view would create a coherence/invalidation burden and the state-entanglement §13's bundle-vs-section split exists to avoid. They are **explicitly documented as winding-lossy**: the log of a versor is multivalued (a rotor by θ and by θ+2π share a holonomy), and that winding information is physically real (it is exactly what the Aharonov–Bohm phase depends on). Therefore the **holonomy is canonical and lossless; the extracted tensor is the lossy convenient view** — never the reverse (§14). Materializing a derived view into a stored `Field` is a *caller's* explicit choice (e.g. a simulation snapshotting a timestep, which then owns that data), never a cache the engine maintains.

- **Nonmetricity `Q` — separate, no holonomy.** A `Field` over 1-cells from inter-node metric variation (the L7 local-metric capability evaluated at an edge's two endpoints), valued in a symmetric bilinear-form difference. It does **not** come from the loop machinery (§15.1: edges rotate via transport → R/T; nodes deform via metric → Q). The loop abstraction therefore never has to carry the metric sector.

**Net:** lossless primitive (connection + loop-holonomy), neutral across graph/complex, potentials-primary, edges-rotate/nodes-deform, no hidden cached state, and one container type (`Field`) for every output.

## 16. Geometric Objects for the Visualization Layer (L11) — Capture Now, Build at L11

A visual sandbox must be able to *show* the geometric structures the engine computes, not just their numeric values. Several real geometric objects are currently either discarded after use or never materialized, because no present layer consumes their *geometry* (the operators need only numbers). This section records them so they are deliberate deferrals, not oversights. **Governing rule:** none of these are built now; each becomes an **optional, derived base/field capability** when L11 (visualization) is designed, exposed by the realizations that can provide it, and **none alters an existing operator** (which keep their single, exact-on-grids definitions). Building viewable geometry ahead of the layer that displays it (and ahead of the still-open UI-host decision, §10.2) would risk building it in a shape the UI cannot use.

**16.1 The dual complex (the motivating case).** DEC's Hodge star conceptually maps a primal k-quantity to the corresponding **dual** (n−k)-cell (face-centers, perpendicular dual edges, vertex-surrounding dual faces). On a structured `GridBase` the **diagonal Hodge is exact** — it computes the correct number — but it *discards where that number lives* (the dual cell). That geometric location is exactly what a user wants to see (e.g. a primal-edge voltage and the dual-edge current threading perpendicular through it). Decision: the **Hodge operator stays single and exact-on-grids** (L8.2); the **dual complex becomes a separate optional geometric capability** (`dual_complex(b)` + a trait, returning dual cell positions + incidence) provided at L11 by bases that can build it (structured grid: cheap; bare graph: cannot, says so; future unstructured mesh: circumcentric/Galerkin dual, which *also* unlocks the non-diagonal Hodge for irregular meshes). The dual is geometry, not a second operator — there is no approximate-vs-exact operator fork.

**16.2 Swept inventory — other viewable objects we compute or could derive.** A deliberate pass for structures the viewer will want, each to arrive as an optional derived capability at L11:

- **Tangent/cotangent frames (the vielbein/coframe).** Where a base carries a metric/frame, the local basis vectors per cell are directly viewable (the "rods" of the local geometry). Natural to show as per-cell glyphs.
- **The metric as an ellipsoid/indicatrix per cell.** The local bilinear form (§13 metric capability) visualizes as a unit-ball ellipsoid (Euclidean) or hyperboloid (Lorentzian) — the most direct way to *see* signature and metric variation, and to see **nonmetricity** as the ellipsoid *changing* cell-to-cell (the "ruler deforming," §15.2).
- **Transport / holonomy as motion.** The connection (§15.2) is viewable by animating parallel transport of a frame along edges; **holonomy** by showing the frame's failure to return around a loop — the single most intuition-building view for curvature/torsion, and the natural picture for Aharonov–Bohm (the accumulated phase around a loop). Curvature `R` (bivector per 2-cell) renders as an oriented rotation-plane glyph; torsion `T` (vector) as a closure-gap arrow.
- **Multivector field glyphs by grade.** A `Field`'s values want grade-appropriate glyphs: 0 = scalar (color/height), 1 = vector (arrow), 2 = bivector (oriented plane/disk), 3 = trivector (oriented volume), pseudoscalar (sign/handedness). The EM bivector `F` is the headline case. This is the general "render a Field" primitive L11 needs.
- **Orientation / handedness.** The incidence-sign orientation (§13 obligation 2) and the pseudoscalar's sign are viewable as consistent cell orientation — needed to read `d`, `⋆`, and chirality correctly.
- **Cell complex itself + grading.** The primal complex with cells colored/filtered by grade is the base canvas everything else draws on; trivially available from L7 obligation 1.
- **Loops/cycles as first-class drawable paths.** Since holonomy is loop-primitive (§15.2), an arbitrary cycle (especially on a `GraphBase` with no faces) is a viewable object in its own right — the user will want to *draw a loop* and read its holonomy. This is also the interactive-probe primitive.
- **Geodesics / integral curves and flows.** Once L8.1 transport and (later) dynamics exist, integral curves of a vector field and geodesics of a connection are viewable paths — the standard "flow" visualization.
- **Spectra / harmonic structure (later).** The Hodge–Laplacian's kernel (harmonic fields) and spectrum are viewable once L8.2 + an eigensolver exist — the natural way to *see* topology (de Rham/Hodge) in field data; pairs with the §7 persistent-homology analysis tool.

**16.3 Discipline.** Each item above is logged so its current absence is a *recorded deferral*. At L11 they are added as optional capabilities, gated per base, never as changes to L7/L8 operators or the `Field` contract. The build-criterion (§4) still applies: a viewable object enters only when L11 actually renders it, and the UI-host decision (§10.2) is settled first, since it determines the shape the geometry must take.

---

*End Rev 1.6. Amend deliberately; this file is the tie-breaker.*
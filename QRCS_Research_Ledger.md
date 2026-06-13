# QRCS Research Ledger

**Companion to:** `DESIGN.md` (which governs the *engine*; this document governs *QRCS theory claims*).
**Status:** Living document · Rev 1.1 · 2026-06-08 — *Rev 1.1: Experiment 1 RUN; §8 updated with results, analytic confirmation of both failure maxima, the no-topology caveat, and the pre-registration-vs-PM-prior note. §9's "claimable after Experiments" first item is now claimable.*
**Purpose:** Record what the TensorCAD project has established, dissolved, constrained, or left open about QRCS — with an explicit status label on every claim — so that progress toward a paper is cumulative and honest rather than vibes.

**Status labels (use them; they are the discipline):**
- **[DICTIONARY]** — an exact mapping between QRCS vocabulary and established physics. Valuable; not a derivation.
- **[DISSOLVED]** — a question that turned out to be a pseudo-problem; what remains of it is stated.
- **[SETTLED]** — an architectural/structural finding, now load-bearing.
- **[CONSTRAINED]** — an idea that survives only inside stated empirical bounds.
- **[OPEN-DEBT]** — a thing the theory *owes*; named precisely so it can be paid or defaulted on, not forgotten.
- **[EXPERIMENT-PENDING]** — registered, specified, falsification conditions pre-stated, not yet run.

---

## 1. The compute-budget ↔ four-momentum dictionary [DICTIONARY — exact]

Per-pattern, per-tick: total compute ~ `E`; motion-compute ~ `pc`; internal-phase-compute ~ `mc²`. Then the internal-phase *fraction* is `mc²/E = 1/γ = dτ/dt` — the exact relativistic proper-time rate, not an approximation. Photon limit (m = 0: all budget to motion, no internal clock) and rest limit (p = 0: all budget internal) both land correctly. The "nothing actually moves — the pattern re-instantiates neighbor-ward" reading is sound (glider-in-Life; QFT's own particles-as-propagating-excitations).

**Correction adopted (Rev 2.7):** "fixed allotment" holds only for **free** patterns — `E` is frame-dependent, so accelerating a pattern *is* feeding it compute. Consequence worth keeping: **force = a compute-transfer channel between patterns**; work done = budget increased. Cleaner than the original phrasing and required for consistency.

## 2. The quadrature debt [OPEN-DEBT — the central one]

`E² = (pc)² + (mc²)²` is a **norm**, not a budget split. A split (parts summing to the whole) gives `E = pc + mc²`, which is empirically wrong (wrong dilation curve). The sum-of-squares *is* the Minkowski geometry of four-momentum. Therefore: the dictionary's "why quadrature?" is **identical** to "why is the substrate's effective geometry Lorentzian?" — the famous open problem of every discrete-substrate model. The dictionary *relocates* the hard problem to a single, precisely-stated structural fact the substrate owes (compute-rates composing as a norm); it does not pay it. **Time dilation is the easy half**; relativity of simultaneity from a preferred-frame (global-tick) substrate is the hard half, and remains fully open. No QRCS paper may claim Lorentz invariance "emerges" until this debt is paid by derivation, not insertion (see §8).

## 3. Mass as internal clock; the c² question [DISSOLVED → reframed]

The square on c is a **unit-conversion artifact**: mass/length/time were assigned unrelated human units, so converting mass to energy needs a velocity², and the invariant speed is the only canonical one. In substrate units (c = one hop per tick = 1): `E² = p² + m²`, no c anywhere. What remains after the dissolution is *better* than the question: **mass is the internal clock rate** — `ω₀ = mc²/ℏ`, the Compton frequency, de Broglie's original "internal clock" whose boost-transformation bookkeeping is exactly the dilation accounting QRCS performs. In the substrate: *internal phase-steps per tick ∝ m*; `E = mc²` becomes the **definition** of mass, not a mystery about light.

## 4. Time architecture [SETTLED — and ordinary, which is good news]

QRCS's global Planck tick = the simulation loop parameter (coordinate time); observed/proper time = internal phase = a **derived per-pattern observable**. This is the standard architecture of numerical GR (integrate in coordinate time, compute proper time along worldlines), not an exotic paradigm. Engine-side this is `DESIGN.md` §18; nothing in the engine is QRCS-specific. The theory's time model costs the toolbox nothing — and gains the standing caveat of §2 above.

## 5. The dielectric-vacuum / variable-c proposal [CONSTRAINED + UNIFIED]

**The unification (the genuinely new finding):** a vacuum whose "refractive scale" varies spatially is a **conformal deformation** `g → K(x)·g` — which is exactly the **trace sector of QRCS's own nonmetricity** (the Weyl vector `Q_μ`). The dielectric-vacuum idea was not a new thread; it is the conformal special case of the generative sector the MAG framework already formalizes. Two threads of the program are one thread.

**The exact part (safe ground):** by the Plebanski correspondence, Maxwell on curved spacetime ≡ Maxwell on flat space in a bianisotropic medium with metric-determined ε, μ (the mathematical basis of transformation optics). "Vacuum as medium" is an exact *re-description* of metric variation — gravity kinematics in medium language (Shapiro delay = the medium slowing light). Lineage: Wilson 1921, Dicke 1957, Puthoff (polarizable-vacuum models). Known limitation: PV models capture weak-field *kinematics*, strain or fail at full GR *dynamics* (GW polarizations, frame-dragging, strong field).

**The fork (where the discipline bites):** if rulers and clocks co-vary with the modified vacuum **exactly**, "variable c" is a gauge choice — pure re-description, the physical content is ordinary gravity, nothing new. If the co-variation is **inexact** — different systems respond differently — it is new physics and already ferociously constrained: this is Einstein's **second clock effect** that killed Weyl's 1918 theory (sharp spectral lines forbid history-dependent units), plus modern variation-of-constants bounds (atomic clocks `α̇/α ≲ 10⁻¹⁷/yr`, Oklo, Eötvös ~10⁻¹⁵). QRCS's "Weyl-consistent" formulation must locate itself on this fork explicitly (integrable Weyl structure, or matter coupling only to invariant combinations).

**Supporting empirical facts:** GW170817 pinned `|v_gw − c|/c ≲ 10⁻¹⁵` — two unrelated massless fields share one speed, which is what "c belongs to the substrate" predicts (and what QRCS itself predicts: c = the graph's hop rate, inherited by everything), and what "c is an EM-tunable property" does not. EM *does* source gravity trivially (stress-energy), but at `G/c⁴ ≈ 8×10⁻⁴⁵` — lab fields are ~40 orders short of measurably deforming the vacuum; QED vacuum effects (Euler–Heisenberg, Scharnhorst) modify propagation *through* regions at parts in 10²⁰⁺ and never move the causal c. Any stronger EM↔vacuum coupling is a new-physics postulate facing the gauntlet above.

## 6. Structural findings now load-bearing in the engine [SETTLED]

Recorded here because they originated as QRCS questions and are now verified engine architecture: **potentials are primary, fields derived** (Aharonov–Bohm = holonomy of the connection; `DESIGN.md` §14, dynamically realized in §19's potential-first state). **Edges rotate, nodes deform** — R/T from transport holonomy, Q from inter-node metric variation; versor conjugation *cannot* carry nonmetricity (it preserves the quadratic form), so the relational/generative split is structurally forced, not stylistic. **EM is a one-sided gauge transport on its own bundle**, distinct from two-sided geometric transport; unified-Clifford-fibre EM was rejected as *nominal* unification (every route — pseudoscalar phase, Kaluza–Klein, bespoke action — reintroduces the distinction in costume) and remains a model-layer construction one may still explore *on* the engine.

## 7. Standing open problems of the MAG framework [OPEN-DEBT — from external review, accepted]

(1) The quadratic nonmetricity action is incomplete: **five** parity-even invariants exist, not three; ghost-freedom requires specific relations among all five, not deletion of two. (2) **Dead torsion**: T has ontological billing but no Lagrangian terms, so its own EOM kills it — include `T²` invariants or justify the suppression. (3) **Fermion decoupling**: minimally-coupled spinors see only the antisymmetric connection — Q drops out of the Dirac operator; the generative sector touches matter only via explicit hypermomentum coupling (`Q_μ ψ̄γ^μψ`-type), which must be added and constrained. These are *dynamics* debts; none touches the engine. The sandbox's job is to make them testable (e.g., transport a spinor and watch Q drop out), not to pre-solve them.

## 8. Registered experiments [EXPERIMENT-PENDING] — with pre-registered readings

**Experiment 1 — the dilation curve [RUN — Rev 1.1; results committed at `scripts/experiments/qrcs_dilation/`].** A walker on a 12-node cycle `GraphBase`, T = 10,000 ticks, deterministic accumulator motion, three pre-registered allocation rules. **Result, exactly as pre-registered:** PYTHAGOREAN tracks `1/γ` to float roundoff (max |dev| 2.3×10⁻¹³, equal to the loop self-check error — i.e. zero beyond plumbing); LINEAR fails with max |dev| 0.41414; QUADRATIC (control) fails with max |dev| 0.24589, sitting *below* target. **Analytic confirmation (post-hoc, strengthens the result):** both failure maxima land on closed-form values — linear's worst case is `√2 − 1 ≈ 0.41421` at `v = 1/√2`; quadratic's is exactly `¼` at `v = √3/2` — the measured maxima are these, sampled adjacent to their peaks. **Permitted claim (Ledger §9 shape, as reported):** Lorentz dilation discriminates among allocation laws and selects norm-composition; dilation is *reproduced under an inserted law, not derived*; no emergence is claimed. **Recorded caveat:** the graph plays no dynamical role in this experiment — the phase depends only on the rule; "on a GraphBase" must not be read as topology mattering. **Registered follow-up unchanged:** deriving the Pythagorean rule from substrate dynamics is the §2 debt. (A PM expected-outcome error — predicting the control would overshoot when `γ⁻² ≤ γ⁻¹` puts it below — was caught by the agent and the measured direction reported, not steered: the pre-registration discipline functioning against the experimenters' own priors, which is its job.)

**Experiment 2 — metric-gradient refraction (needs the variable-Hodge-weights mini-phase first).** Evolve an L10 wave packet across a region of graded per-cell metric (entering via the `_hodge_weight` seam — the single documented entry point for non-unit volumes); observe refraction and slowing: a **Shapiro-delay analog on the grid**, demonstrating trace-Q-as-medium *kinematics* in the sandbox. **Pre-registered reading:** success demonstrates the *exact-re-description* half of §5 (gravity kinematics in medium language) — it does **not** demonstrate EM control of gravity, vacuum engineering, or any inexact-co-variation effect; those remain on the constrained fork.

## 9. Paper-readiness assessment [the honest one]

**Defensibly claimable today (no further results needed):** a well-posed computational framework for metric-affine + gauge transport on discrete substrates, externally verified to an unusual standard; the exact budget↔four-momentum dictionary with `1/γ` as an energy fraction; mass-as-Compton-clock as the substrate-native reading of `E=mc²`; the trace-Q ↔ dielectric-vacuum unification with its fork and constraints; a precise ledger of what the theory owes (§2, §7).

**Claimable after Experiments 1–2:** the discriminative dilation result (Lorentz kinematics selects the norm law among allocation rules); medium-language gravity kinematics demonstrated on the substrate.

**NOT claimable until the §2 debt is paid:** emergence of Lorentz invariance; "time dilation explained" (it is *reproduced under an inserted law*, which is a different sentence); any unification of gravity and EM beyond the exact-re-description kinematics.

**Anti-amplifier rule (binding on the paper):** every claim carries its status label; no result obtained by inserting the answer is presented as deriving it. The referee this document simulates is the one the paper will actually face.

---

*End Rev 1.1. This ledger is the tie-breaker for QRCS claims, as DESIGN.md is for the engine. Amend deliberately.*
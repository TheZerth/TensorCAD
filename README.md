# TensorCAD

**Engine Package:** `Tensorsmith`  
**Author:** Kainaan Riordan

> A focused, interactive environment for exploring algebras of geometric quantities and the calculus of how those quantities change.

TensorCAD is an exact computational multilinear algebra engine. Built around the free tensor algebra $T(V)$ and its quotients (symmetric, exterior, Clifford), the engine acts as a neutral instrument for testing models across varied base spaces; from smooth manifolds to discrete cell complexes.

It is designed to model **emergent structures** natively. By isolating base space transport from a derived metric, the system captures phenomena like **nonmetricity** within general metric-affine geometry directly through its foundational operations.

---

## 🏛 Architecture: General at the Bottom

The core invariant of the system is **one product, many normalizers**. Every named algebra is a quotient of the free tensor algebra $T(V)$ by some two-sided ideal.

* **Exact-First Evaluation:** The default exact ring is `Rational{BigInt}`, allowing precise calculation of symmetries, antisymmetries, and metric inverses without floating-point artifacts.
* **Generic over $R$:** The scalar ring $R$ is entirely generic. Plug in exact rationals, `Symbolics.Num` for symbolic coefficients (a weak dependency — the extension activates on `using Tensorsmith, Symbolics`), or Dual Numbers (`Cl(0,0,1)`) for exact forward-mode automatic differentiation.
* **The BaseSpace Contract (L7):** The engine does not impose a single ontology. It provides an interface supporting smooth charted manifolds, DEC cell complexes, and Clifford-bundles-over-graph equally. This enables the rigorous modeling of complex affine connections, where rotational holonomy (curvature), translational closure failure (torsion), and magnitude drift (nonmetricity) are independently measurable and cleanly separated.
* **Potentials are Primary (L8):** The connection (the potential) is the stored object; field strengths are derived from it by holonomy and differentiation — never the reverse. The discrete exterior derivative `d` is metric-free and works on every base; the Hodge stack (`⋆`, `δ`, `Δ`) is gated behind a real, externally verified dual correspondence.
* **Typed Operator Calculus (L9):** The equation blackboard turns the verified operators into a symbolic layer — equations as first-class, typechecked objects, with a tiny rewrite registry whose every rule is a verified identity.

## 🗜 Layer Map

| Layer | System | Description | Status |
| :--- | :--- | :--- | :--- |
| **L0** | **Scalar Ring** | Exact `Rational{BigInt}` default; Dual numbers for AD; `Symbolics.Num`. | ✅ |
| **L1** | **Vector Space** | `VectorSpace` $V$ and dual $V^*$. | ✅ |
| **L2** | **Free Tensor** | $T(V)$ with non-commutative concatenation. | ✅ |
| **L3** | **Quotients** | Symmetric $Sym(V)$, Exterior $\Lambda(V)$, and Clifford $Cl(V,g)$ algebras. | ✅ |
| **L4** | **Calculus** | Contraction, trace, musical isomorphisms, exact inverse metric. | ✅ |
| **L5** | **GA Suite** | Involutions, grade-projected products, duals, rotors, and inverses. | ✅ |
| **L6** | **Maps & Equivalence** | Projections/sections, symbol maps; cross-system equivalence engine. | Partial |
| **L7** | **Base Space** | Manifold / Grid / Graph realizations with derived metric capabilities. | ✅ |
| **L8** | **Differential Operators** | `d`/grad/curl; `∇`, holonomy, curvature/torsion/nonmetricity; Hodge `⋆`/`δ`/`Δ` with the real cubical dual correspondence; grade-crossing Maxwell. Externally verified. | ✅ |
| **L9** | **Equation Blackboard** | Typed operator calculus: field placeholders, typechecked `Equation`s, verified-identity rewriting, evaluation against the real operators. | ✅ |
| **L10** | **Simulation Engine** | DEC time evolution, structure-preserving integrators, compile-to-fast. | Planned |
| **L11** | **Visualization / UI** | Multivector/field rendering; interactive exploration. | Planned |

## 🧮 The Equation Blackboard (L9)

Equations are written in Julia itself — multiple dispatch is the parser; LaTeX is output only. Every expression node carries and propagates **(grade, residence, base)**, so malformed equations fail at construction, not at evaluation.

```julia
using Tensorsmith

grid = GridBase(1, 1)
A = FieldVar(:A, grid, 1)            # symbolic grade-1 potential
J = FieldVar(:J, grid, 1)

maxwell = Equation(δ(grid, d(A)), J) # typechecked on construction

simplify(d(d(A)))                    # → typed zero: Bianchi falls out of the registry
simplify(⋆(grid, ⋆(grid, A)))        # → (-1)^(k(n-k)+q) · A, the verified sign law

# Bind the placeholders and check numerically with the same verified operators:
check(maxwell; bindings = Dict(:A => A_field, :J => J_field))  # exact residual test
latex(maxwell)                       # LaTeX rendering (output only — no parsing)
```

The rewrite registry is **data** (`DEFAULT_RULES`, extensible via `register_rule!`) and the rewriting strategy is a swappable function — deliberately small: every shipped rule encodes an identity the engine has verified (`d∘d = 0`, `δ∘δ = 0`, the `⋆⋆` sign law, linearity). The definitional substitutions `δ ↔ ±⋆d⋆` and `Δ ↔ dδ + δd` are explicit, user-invoked expansions via `expand`.

---

## ⚙️ Commands & Development

All commands run from the repository root (`TensorCAD`) with the local project environment active.

### Run the Full Test Suite
Executes the full suite and resolves test target dependencies (`Test` + `Symbolics`).

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Faster Iteration
Runs the whole suite in a session without `Pkg.test`'s sandbox. Symbolic test blocks skip gracefully when `Symbolics` is not available in the environment.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'
```

### Run a Single Test File
Each `test/test_*.jl` assumes `using Test, Tensorsmith` are already in scope:

```bash
julia --project=. -e 'using Test, Tensorsmith; include("test/test_clifford.jl")'
```

### Notes
* `Symbolics` is a **weak dependency** (`[weakdeps]` + package extension): the core has zero non-stdlib runtime dependencies, and the extension loads automatically on `using Tensorsmith, Symbolics`.
* `DESIGN.md` is the project charter and tie-breaker — what is built, why, and what is deliberately not built.
* `test/fixtures/` holds optional JSON product tables from GASmith (an independent C++ engine) for blade-by-blade Clifford cross-checks; tests skip gracefully when fixtures are absent.

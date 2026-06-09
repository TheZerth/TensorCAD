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
* **Generic over $R$:** The scalar ring $R$ is entirely generic. Plug in exact rationals, `Symbolics.Num` for symbolic canonicalization, or Dual Numbers (`Cl(0,0,1)`) for exact forward-mode automatic differentiation.
* **The BaseSpace Contract (L7):** The engine does not impose a single ontology. It provides an interface supporting smooth charted manifolds, DEC cell complexes, and Clifford-bundles-over-graph equally. This enables the rigorous modeling of complex affine connections, where rotational holonomy (curvature), translational closure failure (torsion), and magnitude drift (nonmetricity) are independently measurable and cleanly separated.

## 🗜 Layer Map

| Layer | System | Description |
| :--- | :--- | :--- |
| **L0** | **Scalar Ring** | Exact `Rational{BigInt}` default; Dual numbers for AD; `Symbolics.Num`. |
| **L1** | **Vector Space** | `VectorSpace` $V$ and dual $V^*$. |
| **L2** | **Free Tensor** | $T(V)$ with non-commutative concatenation. |
| **L3** | **Quotients** | Symmetric $Sym(V)$, Exterior $\Lambda(V)$, and Clifford $Cl(V,g)$ algebras. |
| **L4** | **Calculus** | Contraction, trace, musical isomorphisms, exact inverse metric. |
| **L5** | **GA Suite** | Involutions, Hodge star, rotors, and inverses. |
| **L7** | **Base Space** | Manifold / Grid / Graph realizations with derived metric capabilities. |

---

## ⚙️ Commands & Development

All commands run from the repository root (`TensorCAD`) with the local project environment active.

### Run the Full Test Suite
Executes the full suite and resolves test target dependencies (`Test` + `Symbolics`).

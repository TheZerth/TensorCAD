# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tensorsmith is a Julia package for exact computational multilinear algebra: the free tensor algebra T(V) and its quotients (symmetric, exterior, Clifford), plus the linear maps between them. The package name in `Project.toml` is `Tensorsmith`; the repo directory is `TensorCAD`.

## Commands

All commands run from the repo root with the local project environment active.

```bash
# Run the full test suite (resolves the test target deps: Test + Symbolics)
julia --project=. -e 'using Pkg; Pkg.test()'

# Faster iteration: run the whole suite in a session without Pkg.test's sandbox
julia --project=. -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'
```

Running a **single test file** standalone: each `test/test_*.jl` assumes `using Test, Tensorsmith` are already in scope (they are `include`d by `runtests.jl`, not self-contained). To run one in isolation:

```bash
julia --project=. -e 'using Test, Tensorsmith; include("test/test_clifford.jl")'
```

The symbolic tests (`test/test_symbolics.jl`) additionally need `using Symbolics`, which triggers the package extension.

## Architecture

### The core invariant: one product, many normalizers

Every named algebra is `T(V)/I` for some two-sided ideal `I`. Rather than implementing a separate product per algebra, the codebase implements **concatenation once** and varies a `normalize` rule per algebra:

```
a ·_A b  =  Σ_{I,J}  c_I · d_J · normalize_A(I ∥ J)
```

`normalize` dispatches on a singleton algebra tag type (`FreeAlgebra`, `SymmetricAlgebra`, `ExteriorAlgebra`), so adding an algebra means adding a `normalize` method — no existing product code changes. Read `src/quotient_algebras.jl` first; its header table maps each algebra to its ideal and canonical multi-index form (non-decreasing for symmetric, strictly-increasing+sign for exterior).

### Generic over a scalar ring R

Every type is parametric in a scalar ring `R` satisfying a duck-typed interface (`zero`, `one`, `+ - *`, `==`, `iszero`, `R(::Integer)` — see the header of `src/scalar_ring.jl`). The default exact ring is `Rational{BigInt}` (exposed as `ExactRing`), chosen so the package has **zero non-stdlib dependencies** in the core. This is a deliberate design constraint — do not add runtime dependencies to `[deps]`. `Rational{BigInt}` contains ℚ exactly, which the symmetrize/antisymmetrize maps need (they divide by k!).

### Representation

Elements are sparse `Dict{key, R}` mapping a multi-index to a coefficient. The empty key is the grade-0/scalar component; zero coefficients are pruned. T(V) and the quotient algebras key on `Vector{Int}` (repetition allowed, non-commutative concatenation); the Phase-6 `MixedTensor` keys on `Vector{Tuple{Int,Variance}}` to track per-slot variance. The header of `src/free_tensor.jl` explains why a bitmask blade representation is *not* used at the base layer (it can't represent `e_i⊗e_i` and implicitly imposes antisymmetry).

All element types subtype `AbstractTensorElement{R}` (Phase 6, `src/abstract_tensor.jl`), which lifts the shared interface — `iszero`, `==`, `hash`, `grade`, `grades`, `homogeneous_component` — to generic methods. Each concrete type provides three hooks: `_eq_key(t)` (the value that, with `terms`, defines identity: `.space` for Free/Algebra/Mixed, `.metric` for Clifford), `base_space(t)`, and `_rebuild(t, dict)`.

### Module layout (load order in `src/Tensorsmith.jl`)

1. `scalar_ring.jl` — ring interface + `ExactRing` default.
2. `abstract_tensor.jl` — `AbstractTensorElement{R}` supertype and the lifted generic grading/equality interface (with the `_eq_key`/`base_space`/`_rebuild` hooks).
3. `vector_space.jl` — `VectorSpace`: rank-n free module with cosmetic basis labels (no metric). **Equality is dimension-only** — labels are display metadata and do not gate compatibility.
4. `free_tensor.jl` — `FreeTensor{R}`, the base T(V) with concatenation product.
5. `quotient_algebras.jl` — algebra tag types, `normalize`, and the unified `AlgebraTensor{A,R}` (Phase 2) for symmetric/exterior.
6. `metric.jl` — `Metric{R}` symmetric bilinear form; `signature_metric`/`diagonal_metric` constructors.
7. `clifford.jl` — `CliffordTensor{R}`. **Note:** this is a standalone type, *not* `AlgebraTensor`, because the metric is a runtime instance and Julia cannot use arbitrary structs as type parameters. Same `Dict{Vector{Int},R}` storage as exterior; Clifford with `Q=0` must degenerate exactly to exterior (a mandatory test).
8. `algebra_maps.jl` — inter-algebra maps (`project_ext`, `project_sym`, `project_cl`, `symmetrize`, `antisymmetrize`, `ext_to_cl`/`cl_to_ext`). The roundtrip and ring-homomorphism invariants are listed in the file header and verified by tests.
9. `tensor_calculus.jl` — Phase 6 tensor-calculus core: `Variance` (`Up`/`Down`) tags, the mixed-variance `MixedTensor{R}` (a *sibling* of `FreeTensor`; all-`Up` reduces to T(V) exactly via `MixedTensor(::FreeTensor)`/`as_free_tensor`), the dual space (covectors are `Down`-variance slots), the metric-free `contract`/`trace`, and the musical isomorphisms `lower`/`raise`. `raise` needs `inverse_metric`, computed by an **exact cofactor/adjugate inverse** (no `LinearAlgebra` dep — keeps the zero-non-stdlib invariant and works symbolically without pivot zero-tests); a degenerate metric (`r>0`, `det=0`) makes `raise` throw an explicit `ArgumentError`.
10. `symbolics_stubs.jl` — empty function declarations (`symbolic_vars`, `symbolic_element`, `symbolic_clifford_vector`, `symbolic_metric`, `isequal_simplified`) whose methods live in the extension.

### Symbolics as a weak dependency (package extension)

`Symbolics` is in `[weakdeps]`/`[extensions]`, not `[deps]`. The extension `ext/TensorsmithSymbolicsExt.jl` activates automatically on `using Tensorsmith, Symbolics` and has zero impact on users who never load it. The core algebra is already fully generic over `R`, so `Symbolics.Num` slots in as another concrete `R`; the extension only adds the `contains_rationals(::Type{Num}) = true` trait (gating symmetrize/antisymmetrize) plus ergonomic constructors. When adding symbolic-only functionality, declare a stub `function f end` in `symbolics_stubs.jl` and implement `Tensorsmith.f(...)` in the extension.

## Conventions

- Multi-indices are **1-based** `Vector{Int}`; grade-0 is `Int[]`. Mixed-variance keys are `Vector{Tuple{Int,Variance}}`; the empty vector is the scalar.
- Source files lead with a long header comment stating the mathematical contract (ideal, canonical form, grade dimensions) and design rationale. Match this style — the math contract belongs in the file, not just the docstring.
- Phase numbers (0–6) appear throughout comments and mark the historical build order; they are not separate code paths.
- Symbolic (`R = Symbolics.Num`) equality must use `isequal_simplified`, never `==`: comparing `Dict{...,Num}` invokes `Num == Num`, which is not a `Bool`. Exact rings (`Rational{BigInt}`) use `==`.

## GASmith cross-check fixtures

`test/fixtures/` holds JSON product tables exported from GASmith (an independent C++ implementation) to validate Clifford normalization blade-by-blade. Tests skip gracefully when a fixture file is absent, so CI passes without them. The format and export procedure are documented in `test/fixtures/README.md`.

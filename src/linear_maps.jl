# ── Linear maps and outermorphisms ────────────────────────────────────────────
#
# MATHEMATICAL CONTRACT
#
# A LinearMap{R} is a linear endomorphism f : V → V of one VectorSpace V.  It is
# deliberately square-only for this backend tidy-up: rectangular maps V → W and
# bundle/base-space maps are later layers, not part of this operation.
#
# MATRIX CONVENTION
#
# The stored matrix A uses the standard column-image convention:
#
#     f(eᵢ) = Σⱼ A[j,i] eⱼ
#
# Therefore the component column of a vector transforms as v′ = A*v, and
# composition (f ∘ g)(v) = f(g(v)) has matrix A_f * A_g.  We implement the tiny
# matrix products by hand rather than depending on LinearAlgebra.
#
# OUTERMORPHISM
#
# The outermorphism is the unique grade-preserving extension of f:
#   - on T(V): the tensor functor f(v₁⊗⋯⊗vₖ) = f(v₁)⊗⋯⊗f(vₖ)
#   - on Λ(V): f(v₁∧⋯∧vₖ) = f(v₁)∧⋯∧f(vₖ)
#   - on Cl(V,g): the same blade formula, using the exterior/wedge product
#     inside the Clifford algebra, not the geometric product
#   - on MixedTensor: Up slots transform by A, Down slots by (A⁻¹)ᵀ
#
# This file is a documented dispatch surface: adding a new element type's action
# should be a new outermorphism method, not a rewrite of existing methods.

"""
    LinearMap{R}

A linear endomorphism `f : V → V` of a [`VectorSpace`](@ref), stored as an
`n×n` matrix over scalar ring `R`.

Matrix convention: column `i` is the image of basis vector `eᵢ`,
`f(eᵢ) = Σⱼ matrix[j,i] eⱼ`.  Equivalently, component columns transform as
`v′ = matrix * v`.  Only endomorphisms are supported in this layer; rectangular
`V → W` maps are intentionally out of scope.

Use [`linear_map`](@ref) and [`identity_map`](@ref) to construct values.
"""
struct LinearMap{R}
    space  :: VectorSpace
    matrix :: Matrix{R}

    function LinearMap{R}(space::VectorSpace, matrix::AbstractMatrix{R}) where R
        n = space.n
        size(matrix, 1) == size(matrix, 2) || throw(ArgumentError(
            "LinearMap matrix must be square; got size $(size(matrix))"))
        size(matrix) == (n, n) || throw(ArgumentError(
            "LinearMap matrix size $(size(matrix)) does not match space dimension $n"))
        new(space, Matrix{R}(matrix))
    end
end

"""
    linear_map(space, R, matrix) -> LinearMap{R}

Construct a linear endomorphism `f : V → V` over scalar ring `R`.

The matrix must be square with size `(space.n, space.n)`.  Column `i` is the
image of `eᵢ`: `f(eᵢ) = Σⱼ matrix[j,i] eⱼ`, so vector component columns are
multiplied as `matrix * v`.
"""
function linear_map(space::VectorSpace, ::Type{R}, matrix::AbstractMatrix) where R
    LinearMap{R}(space, Matrix{R}(matrix))
end

"""
    identity_map(space, R) -> LinearMap{R}

The identity endomorphism of `space` over scalar ring `R`.
"""
function identity_map(space::VectorSpace, ::Type{R}) where R
    A = fill(zero(R), space.n, space.n)
    for i in 1:space.n
        A[i, i] = one(R)
    end
    LinearMap{R}(space, A)
end

function _matrix_isequal(A::AbstractMatrix, B::AbstractMatrix)
    size(A) == size(B) || return false
    for i in axes(A, 1), j in axes(A, 2)
        isequal(A[i, j], B[i, j]) || return false
    end
    true
end

Base.:(==)(f::LinearMap{R}, g::LinearMap{R}) where R =
    f.space == g.space && _matrix_isequal(f.matrix, g.matrix)

Base.hash(f::LinearMap, h::UInt) = hash(f.matrix, hash(f.space, h))

function Base.show(io::IO, f::LinearMap{R}) where R
    print(io, "LinearMap{$R}($(f.space), matrix = $(f.matrix))")
end

function _check_map_space(f::LinearMap, space::VectorSpace, op::Symbol)
    f.space == space || throw(ArgumentError(
        "Cannot $op: map acts on $(f.space), but element lives over $(space)"))
end

function _matmul(A::Matrix{R}, B::Matrix{R}) where R
    size(A, 2) == size(B, 1) || throw(ArgumentError(
        "matrix sizes $(size(A)) and $(size(B)) are not composable"))
    C = fill(zero(R), size(A, 1), size(B, 2))
    for i in 1:size(A, 1), j in 1:size(B, 2)
        acc = zero(R)
        for k in 1:size(A, 2)
            (iszero(A[i, k]) || iszero(B[k, j])) && continue
            acc = acc + A[i, k] * B[k, j]
        end
        C[i, j] = acc
    end
    C
end

"""
    f ∘ g -> LinearMap

Compose two linear maps on the same vector space.  With the column-image
convention, `(f ∘ g).matrix == f.matrix * g.matrix`, i.e. `(f ∘ g)(v) = f(g(v))`.
"""
function Base.:∘(f::LinearMap{R}, g::LinearMap{R}) where R
    _check_map_space(f, g.space, :compose)
    LinearMap{R}(f.space, _matmul(f.matrix, g.matrix))
end

function _acc_vector_key!(result::Dict{Vector{Int}, R}, key::Vector{Int}, coef::R) where R
    iszero(coef) && return
    prev = get(result, key, zero(R))
    val  = prev + coef
    if iszero(val)
        delete!(result, key)
    else
        result[key] = val
    end
end

function _free_basis_image(f::LinearMap{R}, i::Int) where R
    terms = Dict{Vector{Int}, R}()
    for j in 1:f.space.n
        c = f.matrix[j, i]
        iszero(c) && continue
        terms[[j]] = c
    end
    FreeTensor{R}(f.space, terms)
end

function _ext_basis_image(f::LinearMap{R}, i::Int) where R
    terms = Dict{Vector{Int}, R}()
    for j in 1:f.space.n
        c = f.matrix[j, i]
        iszero(c) && continue
        terms[[j]] = c
    end
    AlgebraTensor{ExteriorAlgebra,R}(f.space, terms)
end

function _clifford_basis_image(f::LinearMap{R}, metric::Metric{R}, i::Int) where R
    terms = Dict{Vector{Int}, R}()
    for j in 1:f.space.n
        c = f.matrix[j, i]
        iszero(c) && continue
        terms[[j]] = c
    end
    CliffordTensor{R}(metric, terms)
end

"""
    apply_map(f::LinearMap{R}, v) -> same element family as v

Apply `f` to a grade-1 element representing a vector in `V`, using the documented
matrix convention `v′ = f.matrix * v`.  The zero vector maps to zero.

Methods are provided for grade-1 [`FreeTensor`](@ref),
`AlgebraTensor{ExteriorAlgebra}`, [`CliffordTensor`](@ref), and all-`Up`
grade-1 [`MixedTensor`](@ref).  Non-grade-1 inputs throw `ArgumentError`; higher
tensor powers and covariant slots are handled by [`outermorphism`](@ref).
"""
function apply_map(f::LinearMap{R}, v::FreeTensor{R}) where R
    _check_map_space(f, v.space, :apply_map)
    isempty(v.terms) && return zero(FreeTensor{R}, v.space)
    all(length(idx) == 1 for idx in keys(v.terms)) || throw(ArgumentError(
        "apply_map expects a grade-1 FreeTensor (a vector); got grades $(grades(v))"))

    result = Dict{Vector{Int}, R}()
    for (idx, c) in v.terms
        i = idx[1]
        for j in 1:f.space.n
            aji = f.matrix[j, i]
            iszero(aji) && continue
            _acc_vector_key!(result, [j], c * aji)
        end
    end
    FreeTensor{R}(v.space, result)
end

function apply_map(f::LinearMap{R}, v::AlgebraTensor{ExteriorAlgebra,R}) where R
    _check_map_space(f, v.space, :apply_map)
    isempty(v.terms) && return zero(AlgebraTensor{ExteriorAlgebra,R}, v.space)
    all(length(idx) == 1 for idx in keys(v.terms)) || throw(ArgumentError(
        "apply_map expects a grade-1 Exterior element (a vector); got grades $(grades(v))"))
    outermorphism(f, v)
end

function apply_map(f::LinearMap{R}, v::CliffordTensor{R}) where R
    _check_map_space(f, v.metric.space, :apply_map)
    isempty(v.terms) && return clifford_zero(v.metric)
    all(length(idx) == 1 for idx in keys(v.terms)) || throw(ArgumentError(
        "apply_map expects a grade-1 Clifford element (a vector); got grades $(grades(v))"))
    outermorphism(f, v)
end

function apply_map(f::LinearMap{R}, v::MixedTensor{R}) where R
    _check_map_space(f, v.space, :apply_map)
    isempty(v.terms) && return mixed_zero(v.space, R)
    all(length(slots) == 1 && slots[1][2] === Up() for slots in keys(v.terms)) ||
        throw(ArgumentError(
            "apply_map expects a grade-1 contravariant (all-Up) MixedTensor vector; " *
            "use outermorphism for covariant slots or higher tensors"))
    outermorphism(f, v)
end

"""
    outermorphism(f::LinearMap, A)

The unique grade-preserving extension of a linear map to a tensor or multivector
object, dispatched on `A`'s element type.

Scalars (grade 0) are fixed, and the zero element maps to zero.  Existing
methods implement the tensor functor on [`FreeTensor`](@ref), the exterior
algebra action on `AlgebraTensor{ExteriorAlgebra}`, the same blade action inside
[`CliffordTensor`](@ref) using [`wedge`](@ref), and the variance-aware action on
[`MixedTensor`](@ref).  New element types extend this operation by adding a new
method.
"""
function outermorphism(f::LinearMap{R}, t::FreeTensor{R}) where R
    _check_map_space(f, t.space, :outermorphism)
    isempty(t.terms) && return zero(FreeTensor{R}, t.space)

    acc = zero(FreeTensor{R}, t.space)
    for (idx, c) in t.terms
        term = scalar_element(t.space, c)
        for i in idx
            term = term * _free_basis_image(f, i)
            isempty(term.terms) && break
        end
        acc = acc + term
    end
    acc
end

function outermorphism(f::LinearMap{R}, A::AlgebraTensor{ExteriorAlgebra,R}) where R
    _check_map_space(f, A.space, :outermorphism)
    isempty(A.terms) && return zero(AlgebraTensor{ExteriorAlgebra,R}, A.space)

    acc = zero(AlgebraTensor{ExteriorAlgebra,R}, A.space)
    for (idx, c) in A.terms
        term = ext_scalar(A.space, c)
        for i in idx
            term = term ∧ _ext_basis_image(f, i)
            isempty(term.terms) && break
        end
        acc = acc + term
    end
    acc
end

function outermorphism(f::LinearMap{R}, A::CliffordTensor{R}) where R
    _check_map_space(f, A.metric.space, :outermorphism)
    isempty(A.terms) && return clifford_zero(A.metric)

    acc = clifford_zero(A.metric)
    for (idx, c) in A.terms
        term = clifford_scalar(A.metric, c)
        for i in idx
            term = wedge(term, _clifford_basis_image(f, A.metric, i))
            isempty(term.terms) && break
        end
        acc = acc + term
    end
    acc
end

function _needs_dual_matrix(t::MixedTensor)
    for slots in keys(t.terms), (_, v) in slots
        v === Down() && return true
    end
    false
end

function _inverse_matrix_for_dual_slots(f::LinearMap{R}) where R
    try
        _matrix_inverse(f.matrix)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError(
            "outermorphism on covariant (Down) slots requires an invertible " *
            "LinearMap so the inverse-transpose (f⁻¹)ᵀ exists; the supplied " *
            "map is degenerate/non-invertible (determinant = 0)."))
    end
end

function _slot_matrix_entry(f::LinearMap{R}, invA::Union{Nothing,Matrix{R}},
                            variance::Variance, j::Int, i::Int) where R
    if variance === Up()
        return f.matrix[j, i]
    else
        # Down slots transform by (A⁻¹)ᵀ.  Since invA = A⁻¹, the column-image
        # coefficient for covector eⁱ → Σⱼ (A⁻¹)ᵀ[j,i] eʲ is invA[i,j].
        return invA[i, j]
    end
end

"""
    outermorphism(f::LinearMap, t::MixedTensor) -> MixedTensor

Variance-aware action of `f` on a mixed tensor.  Contravariant (`Up`) slots
transform by `f.matrix`; covariant (`Down`) slots transform by the inverse
transpose `(f⁻¹)ᵀ`, the pull-back action preserving the natural pairing
`⟨eⁱ, eⱼ⟩ = δⁱⱼ`.  If any `Down` slot is present, `f` must be invertible;
otherwise an `ArgumentError` is thrown.

For an all-`Up` `MixedTensor`, this reproduces the [`FreeTensor`](@ref) tensor
functor exactly: `as_free_tensor(outermorphism(f, MixedTensor(t))) ==
outermorphism(f, t)`.
"""
function outermorphism(f::LinearMap{R}, t::MixedTensor{R}) where R
    _check_map_space(f, t.space, :outermorphism)
    isempty(t.terms) && return mixed_zero(t.space, R)

    invA = _needs_dual_matrix(t) ? _inverse_matrix_for_dual_slots(f) : nothing
    result = Dict{MixedIndex, R}()
    for (slots, c) in t.terms
        partial = Dict{MixedIndex, R}(MixedIndex() => c)
        for (i, variance) in slots
            next = Dict{MixedIndex, R}()
            for (prefix, pc) in partial
                for j in 1:f.space.n
                    mji = _slot_matrix_entry(f, invA, variance, j, i)
                    iszero(mji) && continue
                    newslots = MixedIndex(copy(prefix))
                    push!(newslots, (j, variance))
                    _acc!(next, newslots, pc * mji)
                end
            end
            partial = next
            isempty(partial) && break
        end
        for (newslots, coef) in partial
            _acc!(result, newslots, coef)
        end
    end
    MixedTensor{R}(t.space, result)
end

"""
    determinant(f::LinearMap{R}) -> R

The coordinate-free determinant of `f`, computed from the top exterior power:
if `I = e₁∧⋯∧eₙ` is the unit pseudoscalar in `Λⁿ(V)`, then
`outermorphism(f, I) = determinant(f) * I`.

This avoids exporting or depending on `LinearAlgebra.det`; tests assert that it
agrees with the internal exact cofactor determinant `_det(f.matrix)`.
"""
function determinant(f::LinearMap{R}) where R
    top = collect(1:f.space.n)
    I = alg_basis_element(f.space, R, ExteriorAlgebra, top)
    image = outermorphism(f, I)
    get(image.terms, top, zero(R))
end

export LinearMap,
       linear_map, identity_map,
       apply_map, outermorphism, determinant

# ── Phase 4: Inter-algebra maps ───────────────────────────────────────────────
#
# This file implements the linear maps between the algebras built in Phases 1-3.
# The tower of maps is:
#
#   T(V)  --project_ext-->  Lambda(V)  --antisymmetrize-->  T(V)
#   T(V)  --project_sym-->  Sym(V)     --symmetrize------->  T(V)
#   T(V)  --project_cl-->   Cl(V,g)
#
#   Lambda(V)  <--ext_to_cl / cl_to_ext-->  Cl(V,g)   (vector-space iso)
#
# Key invariants (verified by tests):
#   project_ext(antisymmetrize(t)) == t   for all t in Lambda(V)
#   project_sym(symmetrize(t))     == t   for all t in Sym(V)
#   project_ext and project_sym are ring homomorphisms from T(V)
#   cl_to_ext(ext_to_cl(t, g))     == t   for all t in Lambda(V)
#   ext_to_cl(cl_to_ext(t), g)     == t   for all t in Cl(V,g)
#
# Note on rings: antisymmetrize and symmetrize divide by k!, requiring the ring
# to contain Q. For R = Rational{BigInt} this always holds. Pass R = Float64
# only if you accept floating-point rounding in factorial denominators.

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers

# Accumulate coef into result[key], deleting the key if the sum is zero.
function _accumulate!(result::Dict{Vector{Int}, R}, key::Vector{Int}, coef::R) where R
    iszero(coef) && return
    prev = get(result, key, zero(R))
    val  = prev + coef
    if iszero(val)
        delete!(result, key)
    else
        result[key] = val
    end
end

# Generate all permutations of 1:n as Vector{Int}.
# Returns a Vector of length n!, each entry a Vector of length n.
function _all_position_permutations(n::Int) :: Vector{Vector{Int}}
    n == 0 && return [Int[]]
    result = Vector{Vector{Int}}()
    for sub in _all_position_permutations(n - 1)
        for pos in 1:n
            # Insert n at position pos in sub
            new_perm = vcat(sub[1:pos-1], [n], sub[pos:end])
            push!(result, new_perm)
        end
    end
    result
end

# Sign of a permutation (given as a position array): (-1)^(inversion count).
function _perm_sign(perm::Vector{Int}) :: Int
    n   = length(perm)
    sgn = 1
    for i in 1:n-1, j in i+1:n
        perm[i] > perm[j] && (sgn = -sgn)
    end
    sgn
end

# ─────────────────────────────────────────────────────────────────────────────
# Projections  T(V) -> quotient algebra
#
# Each projection is a surjective ring homomorphism.  The implementation simply
# passes the FreeTensor's term dict to the AlgebraTensor / CliffordTensor
# constructor, which applies the appropriate normalization to every term.

"""
    project_ext(t::FreeTensor{R}) -> AlgebraTensor{ExteriorAlgebra, R}

Surjective ring homomorphism pi: T(V) -> Lambda(V).

Every term of t is reduced by the Exterior normalization rule (sort with sign,
zero on repeated index).  Satisfies:

    project_ext(a * b) == project_ext(a) wedge project_ext(b)
"""
function project_ext(t::FreeTensor{R}) where R
    AlgebraTensor{ExteriorAlgebra, R}(t.space, t.terms)
end

"""
    project_sym(t::FreeTensor{R}) -> AlgebraTensor{SymmetricAlgebra, R}

Surjective ring homomorphism pi: T(V) -> Sym(V).

Every term of t is reduced by the Symmetric normalization rule (sort, no sign).
Satisfies:

    project_sym(a * b) == project_sym(a) * project_sym(b)
"""
function project_sym(t::FreeTensor{R}) where R
    AlgebraTensor{SymmetricAlgebra, R}(t.space, t.terms)
end

"""
    project_cl(t::FreeTensor{R}, metric::Metric{R}) -> CliffordTensor{R}

Surjective ring homomorphism pi: T(V) -> Cl(V,g).

Every term of t is reduced by the Clifford normalization rule (sort with sign,
contract equal pairs via the metric).  The space of t and the space of the
metric must agree.
"""
function project_cl(t::FreeTensor{R}, metric::Metric{R}) where R
    t.space == metric.space || throw(ArgumentError(
        "FreeTensor space $(t.space) does not match metric space $(metric.space)"))
    CliffordTensor{R}(metric, t.terms)
end

# ─────────────────────────────────────────────────────────────────────────────
# Sections  quotient algebra -> T(V)
#
# A section is a right-inverse of the corresponding projection:
#   project_ext(antisymmetrize(t)) == t
#   project_sym(symmetrize(t))     == t
#
# Both maps require division by k! and therefore need the ring to contain Q.
# For R = Rational{BigInt} this is always satisfied.

"""
    antisymmetrize(t::AlgebraTensor{ExteriorAlgebra, R}) -> FreeTensor{R}

Linear section Lambda(V) -> T(V):

    e_{i1} wedge ... wedge e_{ik}  |->  (1/k!) * sum_{sigma in S_k} sgn(sigma) * e_{sigma(i1)} ox ... ox e_{sigma(ik)}

The result is the unique antisymmetric lift of t in T(V).

Requires R to contain Q (i.e., contains_rationals(R) must be true).
"""
function antisymmetrize(t::AlgebraTensor{ExteriorAlgebra, R}) where R
    contains_rationals(R) || throw(ArgumentError(
        "antisymmetrize requires R to contain Q. Use R = Rational{BigInt}."))
    V      = t.space
    result = Dict{Vector{Int}, R}()
    for (idx, coef) in t.terms
        k = length(idx)
        # k! computed in R to avoid integer overflow for large k
        k_fact    = foldl((acc, i) -> acc * R(i), 1:k; init = one(R))
        inv_kfact = one(R) / k_fact
        for pos_perm in _all_position_permutations(k)
            sgn     = _perm_sign(pos_perm)
            new_idx = [idx[pos_perm[j]] for j in 1:k]
            _accumulate!(result, new_idx, coef * R(sgn) * inv_kfact)
        end
    end
    FreeTensor{R}(V, result)
end

"""
    symmetrize(t::AlgebraTensor{SymmetricAlgebra, R}) -> FreeTensor{R}

Linear section Sym(V) -> T(V):

    e_{i1} ... e_{ik}  |->  (1/k!) * sum_{sigma in S_k} e_{sigma(i1)} ox ... ox e_{sigma(ik)}

The result is the unique symmetric lift of t in T(V).

Requires R to contain Q (i.e., contains_rationals(R) must be true).
"""
function symmetrize(t::AlgebraTensor{SymmetricAlgebra, R}) where R
    contains_rationals(R) || throw(ArgumentError(
        "symmetrize requires R to contain Q. Use R = Rational{BigInt}."))
    V      = t.space
    result = Dict{Vector{Int}, R}()
    for (idx, coef) in t.terms
        k = length(idx)
        k_fact    = foldl((acc, i) -> acc * R(i), 1:k; init = one(R))
        inv_kfact = one(R) / k_fact
        for pos_perm in _all_position_permutations(k)
            new_idx = [idx[pos_perm[j]] for j in 1:k]
            _accumulate!(result, new_idx, coef * inv_kfact)
        end
    end
    FreeTensor{R}(V, result)
end

# ─────────────────────────────────────────────────────────────────────────────
# Symbol maps  Lambda(V) <--> Cl(V,g)
#
# Both algebras share the same underlying vector space: as graded vector spaces
# both have dimension 2^n with canonical basis indexed by strictly-increasing
# multi-indices of length 0..n.  The symbol maps are the vector-space isomorphism
# that sends canonical basis elements of one algebra to the corresponding basis
# elements of the other.
#
# The maps do NOT preserve multiplication in general -- they are algebra
# isomorphisms only when the metric is identically zero (Q=0 case), in which
# case Cl(V, 0) == Lambda(V) as algebras.

"""
    ext_to_cl(t::AlgebraTensor{ExteriorAlgebra, R}, metric::Metric{R}) -> CliffordTensor{R}

Vector-space isomorphism Lambda(V) -> Cl(V,g).

Sends each canonical basis blade of Lambda(V) to the corresponding canonical
blade of Cl(V,g) with the same coefficient.  Does NOT preserve multiplication
unless metric == zero_metric (the Q=0 case).
"""
function ext_to_cl(t::AlgebraTensor{ExteriorAlgebra, R}, metric::Metric{R}) where R
    t.space == metric.space || throw(ArgumentError(
        "Exterior element space $(t.space) does not match metric space $(metric.space)"))
    # Canonical Exterior indices are strictly increasing, which is also the
    # canonical form for CliffordTensor.  Pass through without re-normalization.
    CliffordTensor{R}(metric, t.terms)
end

"""
    cl_to_ext(t::CliffordTensor{R}) -> AlgebraTensor{ExteriorAlgebra, R}

Vector-space isomorphism Cl(V,g) -> Lambda(V).

Sends each canonical basis blade of Cl(V,g) to the corresponding canonical
blade of Lambda(V) with the same coefficient.  Does NOT preserve multiplication
unless the metric is zero.
"""
function cl_to_ext(t::CliffordTensor{R}) where R
    AlgebraTensor{ExteriorAlgebra, R}(t.metric.space, t.terms)
end

# ─────────────────────────────────────────────────────────────────────────────
export project_ext, project_sym, project_cl,
       antisymmetrize, symmetrize,
       ext_to_cl, cl_to_ext

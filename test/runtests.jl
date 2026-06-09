using Test
using Tensorsmith

@testset "Tensorsmith" begin
    include("test_scalar_ring.jl")
    include("test_vector_space.jl")
    include("test_free_tensor.jl")
    include("test_quotient_algebras.jl")
    include("test_clifford.jl")
    include("test_algebra_maps.jl")
    include("test_tensor_calculus.jl")
    include("test_ga_operations.jl")
    include("test_number_systems.jl")
    include("test_symbolics.jl")
end

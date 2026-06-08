using Test
using Tensorsmith

@testset "Tensorsmith" begin
    include("test_scalar_ring.jl")
    include("test_vector_space.jl")
    include("test_free_tensor.jl")
    include("test_quotient_algebras.jl")
    include("test_clifford.jl")
end

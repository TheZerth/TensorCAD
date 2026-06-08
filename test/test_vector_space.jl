@testset "VectorSpace" begin

    @testset "Default construction" begin
        V = VectorSpace(3)
        @test V.n == 3
        @test V.labels == [:e1, :e2, :e3]

        V1 = VectorSpace(1)
        @test V1.labels == [:e1]

        V0 = VectorSpace(0)
        @test V0.n == 0
        @test V0.labels == Symbol[]
    end

    @testset "Custom labels" begin
        V = VectorSpace(2, [:x, :y])
        @test V.n == 2
        @test V.labels == [:x, :y]

        V4 = VectorSpace(4, [:t, :x, :y, :z])
        @test V4.labels[1] == :t
        @test V4.labels[4] == :z
    end

    @testset "Equality" begin
        @test VectorSpace(3) == VectorSpace(3)
        @test VectorSpace(2) != VectorSpace(3)
        @test VectorSpace(2, [:x, :y]) != VectorSpace(2, [:e1, :e2])
        @test VectorSpace(2, [:e1, :e2]) == VectorSpace(2)
    end

    @testset "Error cases" begin
        @test_throws ArgumentError VectorSpace(-1)
        @test_throws ArgumentError VectorSpace(2, [:x])          # too few labels
        @test_throws ArgumentError VectorSpace(2, [:x, :y, :z]) # too many labels
    end

end

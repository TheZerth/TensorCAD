@testset "ScalarRing" begin

    @testset "ExactRing is Rational{BigInt}" begin
        @test ExactRing === Rational{BigInt}
        @test contains_rationals(ExactRing)
    end

    @testset "contains_rationals" begin
        # Rational types contain ℚ
        @test contains_rationals(Rational{BigInt})
        @test contains_rationals(Rational{Int64})
        @test contains_rationals(Rational{Int32})

        # Non-rational types do not
        @test !contains_rationals(Float64)
        @test !contains_rationals(Float32)
        @test !contains_rationals(Int)
        @test !contains_rationals(BigInt)
        @test !contains_rationals(Complex{Float64})
        @test !contains_rationals(Complex{Rational{BigInt}})   # complex ℚ: not covered yet
    end

end

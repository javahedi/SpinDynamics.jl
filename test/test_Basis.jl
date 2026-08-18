using Test
using SpinDynamics

@testset "Basis validation" begin
    @test_throws ArgumentError build_full_basis(0)
    @test_throws ArgumentError build_full_basis(-1)

    @test_throws ArgumentError build_sector_basis(4, -1)
    @test_throws ArgumentError build_sector_basis(4, 5)

    states, idxmap = build_sector_basis(4, 0)
    @test length(states) == 1
    @test states[1] == UInt64(0)
    @test idxmap[UInt64(0)] == 1

    states, idxmap = build_sector_basis(4, 4)
    @test length(states) == 1
    @test count_ones(states[1]) == 4
end
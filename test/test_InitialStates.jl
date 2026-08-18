using Test
using SpinDynamics



@testset "Initial-state return consistency" begin
    model_full = XXZChain(4)
    model_sector = XXZChain(4; nup=2)

    ψ_full = domain_wall_state(model_full)
    ψ_sector = domain_wall_state(model_sector)

    @test ψ_full isa Vector{Float64}
    @test ψ_sector isa Vector{Float64}

    @test length(ψ_full) == length(model_full.states)
    @test length(ψ_sector) == length(model_sector.states)

    @test sum(abs2, ψ_full) ≈ 1.0
    @test sum(abs2, ψ_sector) ≈ 1.0
end
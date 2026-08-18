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


    L = model_full.L

    ψ_neel_full = neel_state(model_full)

    @test ψ_neel_full isa Vector{Float64}
    @test length(ψ_neel_full) == length(model_full.states)
    @test sum(abs2, ψ_neel_full) ≈ 1.0
    @test count(!iszero, ψ_neel_full) == 1

    s_expected = UInt64(0)
    for i in 0:(L - 1)
        if isodd(i + 1)
            s_expected |= UInt64(1) << i
        end
    end

    @test ψ_neel_full[Int(s_expected) + 1] == 1.0
end


@testset "Polarized state" begin
    model_full = XXZChain(4)

    ψ_up = polarized_state(model_full; up=true)
    ψ_down = polarized_state(model_full; up=false)

    @test ψ_up isa Vector{Float64}
    @test ψ_down isa Vector{Float64}
    @test length(ψ_up) == 16
    @test length(ψ_down) == 16

    @test count(!iszero, ψ_up) == 1
    @test count(!iszero, ψ_down) == 1
    @test sum(abs2, ψ_up) ≈ 1.0
    @test sum(abs2, ψ_down) ≈ 1.0

    # |↑↑↑↑⟩ = 0b1111
    @test ψ_up[16] == 1.0

    # |↓↓↓↓⟩ = 0b0000
    @test ψ_down[1] == 1.0

    model_up = XXZChain(4; nup=4)
    model_down = XXZChain(4; nup=0)
    model_half = XXZChain(4; nup=2)

    @test polarized_state(model_up; up=true) == [1.0]
    @test polarized_state(model_down; up=false) == [1.0]

    @test_throws ArgumentError polarized_state(model_half; up=true)
    @test_throws ArgumentError polarized_state(model_half; up=false)
end



@testset "Polarized state with flips" begin
    model_full = XXZChain(4)

    ψ = polarized_state_with_flips(model_full, [2, 4])

    @test ψ isa Vector{Float64}
    @test length(ψ) == 16
    @test count(!iszero, ψ) == 1
    @test sum(abs2, ψ) ≈ 1.0

    # Start 1111, flip sites 2 and 4 -> 0101
    s_expected = UInt64(0b0101)
    @test ψ[Int(s_expected) + 1] == 1.0

    # Two flipped spins -> nup = 2
    model_sector = XXZChain(4; nup=2)
    ψ_sector = polarized_state_with_flips(model_sector, [2, 4])

    @test ψ_sector isa Vector{Float64}
    @test count(!iszero, ψ_sector) == 1
    @test sum(abs2, ψ_sector) ≈ 1.0

    # Wrong sector
    model_wrong = XXZChain(4; nup=3)
    @test_throws ArgumentError polarized_state_with_flips(
        model_wrong, [2, 4]
    )

    # Invalid sites
    @test_throws ArgumentError polarized_state_with_flips(model_full, [0])
    @test_throws ArgumentError polarized_state_with_flips(model_full, [5])
end
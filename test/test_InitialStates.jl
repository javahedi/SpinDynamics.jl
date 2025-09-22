using Test
using SpinDynamics

# ----------------------------
# Test InitialStates
# ----------------------------
@testset "InitialStates Tests" begin

    L = 4
    nup = 2
    model_sector = build_model(L; nup=nup)
    model_full = build_model(L)

    # ------------------------
    # Domain wall state
    # ------------------------
    ψ_dw_sector = domain_wall_state(model_sector)
    @test isa(ψ_dw_sector, Vector{Float64})
    @test sum(ψ_dw_sector) == 1.0
    # Only one non-zero entry
    @test count(!iszero, ψ_dw_sector) == 1

    s_dw_full = domain_wall_state(model_full)
    @test isa(s_dw_full, UInt64)
    # Domain wall: first nup bits set
    for i in 0:Int(floor(L/2))-1
        @test ((s_dw_full >> i) & 0x1) == 1
    end

    # ------------------------
    # Néel state
    # ------------------------
    ψ_neel_sector = neel_state(model_sector)
    @test isa(ψ_neel_sector, Vector{Float64})
    @test sum(ψ_neel_sector) == 1.0
    @test count(!iszero, ψ_neel_sector) == 1

    s_neel_full = neel_state(model_full)
    @test isa(s_neel_full, UInt64)
    # Check alternating ↑↓ pattern
    for i in 0:L-1
        expected = isodd(i+1) ? 1 : 0
        @test ((s_neel_full >> i) & 0x1) == expected
    end

   
    # ------------------------
    # Polarized state
    # ------------------------
    L = 4
    nup = 4
    model_sector = build_model(L; nup=nup)
    model_full = build_model(L)

    # For sector model, specify up=true to get a valid sector state
    ψ_pol_sector = polarized_state(model_sector; up=true)
    @test sum(ψ_pol_sector) == 1.0  # Only one non-zero entry
    @test count(!iszero, ψ_pol_sector) == 1

    # Full Hilbert space
    s_pol_full = polarized_state(model_full)
    @test isa(s_pol_full, UInt64)
    @test s_pol_full == (UInt64(1) << L) - 1


    

   

end

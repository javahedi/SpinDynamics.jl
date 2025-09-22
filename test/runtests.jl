

using Test
using SpinDynamics

include("test_SpinModel.jl")
include("test_InitialStates.jl")
include("test_Hamiltonian.jl")

#=

@testset "Basis: full Hilbert space" begin
    L = 8
    states, idxmap = build_full_basis(L)

    # dimension check: should be 2^L
    @test length(states) == 1 << L
    @test length(idxmap) == 1 << L

    # idxmap consistency
    for (i, s) in enumerate(states)
        @test idxmap[s] == i
    end
end


@testset "Basis: sector Hilbert space" begin
    L = 8
    nup = 4
    states, idxmap = build_sector_basis(L, nup)

    # dimension check: should be binomial(L, nup)
    @test length(states) == binomial(L, nup)
    @test length(idxmap) == binomial(L, nup)

    # all states should have exactly nup ones
    @test all(count_ones(s) == nup for s in states)

    # idxmap consistency
    for (i, s) in enumerate(states)
        @test idxmap[s] == i
    end

    # cross-check: idxmap covers exactly the states
    @test all(haskey(idxmap, s) for s in states)
end





@testset "Hamiltonian consistency" begin
    # Small system

    L = 6                     # small system for quick tests
    nup = 3
    hopping = [(i,i+1,2.0) for i in 1:(L-1)]
    zz = [(i,i+1,0.0) for i in 1:(L-1)]
    onsite = ones(L)

    p = SpinParams(L, hopping, onsite, zz)


    # --- Full Hilbert space ---
    Nfull = 1 << L
    ψfull = randn(Nfull)
    out_full = zeros(Nfull)
    apply_H_full!(out_full, ψfull, p)


    # --- Restricted sector ---
    states, idxmap = build_sector_basis(L, nup)
    Nsec = length(states)
    ψsec = randn(Nsec)
    out_sec = zeros(Nsec)
    apply_H_sector!(out_sec, ψsec, p, states, idxmap)
    
    # --- Check sector = restriction of full ---
    # Embed ψsec into full vector
    ψembed = zeros(Nfull)
    for (i,s) in enumerate(states)
        ψembed[Int(s)+1] = ψsec[i]
    end
    out_embed = zeros(Nfull)
    apply_H_full!(out_embed, ψembed, p)

    # Compare results: sector vs restricted full
    for (i,s) in enumerate(states)
        @test isapprox(out_sec[i], out_embed[Int(s)+1]; atol=1e-12)
    end
end



# -----------------------------
# Test parameters
# -----------------------------
const L = 6                     # small system for quick tests
const nup = 3
const dt = 0.05                   # time step
const hopping = [(i,i+1,2.0) for i in 1:(L-1)]
const zz = [(i,i+1,0.0) for i in 1:(L-1)]
const onsite = zeros(L)
const tol = 1e-3

# -----------------------------
# Helper function to compare S(q)
# -----------------------------
function compare_Sq(Sq1::Dict, Sq2::Dict; tol=1e-6)
    keys1 = sort(collect(keys(Sq1)))
    keys2 = sort(collect(keys(Sq2)))
    @assert keys1 == keys2 "Momentum grids differ"
    diffs = [abs(Sq1[k] - Sq2[k]) for k in keys1]
    return maximum(diffs) < tol
end


# -----------------------------
# Krylov tests
# -----------------------------
@testset "Krylov module" begin

    # -----------------------------
    # 1) Sector vs full evolution
    # -----------------------------
    mags_sector, Sq_sector = run_krylov_sector(L, nup, hopping, onsite, zz, dt)
    mags_full,   Sq_full   = run_krylov_full(L, hopping, onsite, zz, dt)


    @test all(isapprox.(mags_sector, mags_full; rtol=1e-6, atol=1e-3))  #all(abs.(mags_sector .- mags_full) .< tol)
    @test compare_Sq(Sq_sector, Sq_full; tol=tol)

    
    # -----------------------------
    # 2) Krylov time evolution preserves norm
    # -----------------------------
    states, idxmap = build_sector_basis(L, nup)
    ψ0 = domain_wall_state_sector(L, nup, states, idxmap)
    ψ0c = ComplexF64.(ψ0)
    ψt = krylov_time_evolve(ψ0c, dt, apply_H_sector!, 
                            SpinParams(L,hopping,onsite,zz),  m=30, states=states, idxmap=idxmap)

    @test abs(norm(ψt) - norm(ψ0c)) < tol
    
    
    # -----------------------------
    # 3) Krylov evolution reduces to identity at t=0
    # -----------------------------
    ψt0 = krylov_time_evolve(ψ0c, 0.0, apply_H_sector!, 
                            SpinParams(L,hopping,onsite,zz), m=30, 
                            states=states, idxmap=idxmap)

    @test all(abs.(ψt0 .- ψ0c) .< tol)
    

end



# -----------------------------
# Chebyshev tests
# -----------------------------
@testset "Chebyshev module tests" begin



   # ----------------------------- # 1) Sector vs full evolution # ----------------------------- 
   mags_sector, Sq_sector, Ebounds_full= run_chebyshev_sector(L, nup, hopping, onsite, zz, dt) 
   mags_full, Sq_full, Ebounds_sector = run_chebyshev_full(L, hopping, onsite, zz, dt) 
   
   
  
   @test all(isapprox.(mags_sector, mags_full; rtol=1e-6, atol=1e-3)) # #@test all(abs.(mags_sector .- mags_full) .< tol) 
   @test compare_Sq(Sq_sector, Sq_full; tol=tol)


    
    states, idxmap = build_sector_basis(L, nup)
    ψ0 = domain_wall_state_sector(L, nup, states, idxmap)
    ψ0c = ComplexF64.(ψ0)
    ψt_sector = chebyshev_time_evolve(ψ0c, dt, apply_H_sector!, 
                                      SpinParams(L, hopping, onsite, zz),
                                      n=50, Ebounds=Ebounds_sector, 
                                      states=states, idxmap=idxmap)
    @test abs(norm(ψt_sector) - norm(ψ0c)) < tol

    

    # Full Hilbert space evolution

    ψ0_full = zeros(ComplexF64, 1<<L)
    state_dw = domain_wall_state_full(L)
    ψ0_full[Int(state_dw)+1] = 1.0 + 0im
    ψ0c_full = copy(ψ0_full)
    ψt_full = chebyshev_time_evolve(ψ0c_full, dt, apply_H_full!, 
                                    SpinParams(L, hopping, onsite, zz), 
                                    n=50, Ebounds=Ebounds_full)

    @test abs(norm(ψt_full) - norm(ψ0c_full)) < tol

end


# -----------------------------
# KPM module tests
# -----------------------------
@testset "KPM module tests" begin
    
    ω_range = range(-4.0, 4.0, length=50)

    # Test 1: Basic functionality - Sz-Sz correlation
    @testset "Sz-Sz correlation" begin

        # Should not throw errors
        S_ω = run_kpm_dynamical_sector(L, nup, hopping, onsite, zz, 
                                        ω_range; opA_type_a=:z, opB_type_b=:z, n=100)
        
        # Basic checks
    
        @test all(isfinite.(S_ω))
        @test all(S_ω .>= 0.0)  # Spectral function should be non-negative
        @test maximum(S_ω) > 0.0  # Should have some spectral weight

        println("Sz-Sz correlation test passed: spectrum has $(sum(S_ω)/length(S_ω)) total weight")
    end


    # Test 2: S⁺-S⁻ correlation,,, must be done in full space
    @testset "S⁺-S⁻ correlation" begin
       
        run_kpm_dynamical_full(::Int64, ::Int64, ::Vector{Tuple{Int64, Int64, Float64}}, 
        ::Vector{Float64}, ::Vector{Tuple{Int64, Int64, Float64}}, 
        ::StepRangeLen{Float64, Base.TwicePrecision{Float64},
         Base.TwicePrecision{Float64}, Int64}; opA_type_a::Symbol, 
         opB_type_b::Symbol, n::Int64)

        S_ω = run_kpm_dynamical_full(L, nup, hopping, onsite, zz, ω_range; 
        opA_type_a=:plus, opB_type_b=:plus, n=100)

        @test all(isfinite.(S_ω))
        @test all(S_ω .>= 0.0)  # Spectral function should be non-negative
        @test maximum(S_ω) > 0.0  # Should have some spectral weight

        println("S⁺-S⁻ correlation test passed: spectrum has $(sum(S_ω)/length(S_ω)) total weight")
    end
    #=
    # Test 3: Reusable correlation functions
    @testset "Reusable correlation functions" begin
        L = 6
        nup = 3
        ω_range = range(-4.0, 4.0, length=30)
        
        # Create test system
        p = SpinParams(L, [(1,2,1.0)], zeros(L), [(1,2,0.5)])
        states, idxmap = build_sector_basis(L, nup)
        ψ_gs = domain_wall_state_sector(L, nup, states, idxmap)
        ψ_gs = ComplexF64.(ψ_gs)
        ψ_gs ./= norm(ψ_gs)
        
        # Create reusable functions
        sz1_sz2_correlation = create_dynamical_correlation_function(:z, 1, :z, 2)
        splus_sminus_correlation = create_dynamical_correlation_function(:plus, 1, :minus, 1)
        
        # Test they work
        ω1, result1 = sz1_sz2_correlation(ψ_gs, ω_range, p; n=80, states=states, idxmap=idxmap)
        ω2, result2 = splus_sminus_correlation(ψ_gs, ω_range, p; n=80, states=states, idxmap=idxmap)
        
        @test ω1 == ω_range
        @test ω2 == ω_range
        @test all(isfinite.(result1))
        @test all(isfinite.(result2))
        
        println("Reusable correlation functions test passed")
    end
    
    # Test 4: Operator creation
    @testset "Operator creation" begin
        L = 4
        nup = 2
        states, idxmap = build_sector_basis(L, nup)
        ψ_test = ones(ComplexF64, length(states))
        ψ_test ./= norm(ψ_test)
        p = SpinParams(L, [(1,2,1.0)], zeros(L), [(1,2,0.5)])
        
        # Test all operator types
        operators = [
            create_spin_operator(1, :z),
            create_spin_operator(1, :plus),
            create_spin_operator(1, :minus),
            create_spin_operator(1, :x),
            create_spin_operator(1, :y)
        ]
        
        for op in operators
            result = op(ψ_test, p, states, idxmap)
            @test length(result) == length(ψ_test)
            @test all(isfinite.(result))
        end
        
        println("Operator creation test passed: all operator types work correctly")
    end
    
    # Test 5: Error handling
    @testset "Error handling" begin
        # Test invalid operator type
        @test_throws ArgumentError create_spin_operator(1, :invalid)
        
        # Test out of bounds site
        @test_throws BoundsError create_spin_operator(100, :z)
        
        println("Error handling test passed")
    end
    
    # Test 6: Consistency checks
    @testset "Consistency checks" begin
        L = 4
        nup = 2
        ω_range = range(-3.0, 3.0, length=20)
        
        # Test that same-site Sz-Sz gives reasonable results
        ω1, S1 = run_kpm_dynamical_sector(L=L, nup=nup, hopping=1.0, h=0.0, zz=0.5,
                                        ω_range=ω_range, opA_site=1, opB_site=1, n=80)
        
        # Should be mostly positive with some structure
        @test maximum(S1) > 0.0
        @test sum(S1) > 0.0
        
        println("Consistency checks passed: same-site correlation has $(sum(S1)) total weight")
    end
    
    # Test 7: Full Hilbert space (small system)
    @testset "Full Hilbert space" begin
        L = 4  # Small system for full Hilbert space test
        ω_range = range(-3.0, 3.0, length=20)
        
        ω, S_ω = run_kpm_dynamical_full(L=L, hopping=1.0, h=0.0, zz=0.5,
                                      ω_range=ω_range, opA_site=1, opB_site=2, n=80)
        
        @test length(ω) == length(S_ω)
        @test all(isfinite.(S_ω))
        @test maximum(S_ω) > 0.0
        
        println("Full Hilbert space test passed")
    end
    
    # Test 8: Kernel functions
    @testset "Kernel functions" begin
        # Test Jackson kernel
        n = 10
        kernel = get_jackson_kernel(n)
        @test length(kernel) == n
        @test all(0.0 .<= kernel .<= 1.0)
        @test kernel[1] ≈ 1.0 atol=1e-10  # First coefficient should be 1
        
        # Test Chebyshev series evaluation
        μ_n = zeros(5)
        μ_n[1] = 1.0  # Only constant term
        result = evaluate_chebyshev_series(μ_n, 0.5, 1.0)
        @test isfinite(result)
        
        # Test edge case
        edge_result = evaluate_chebyshev_series(μ_n, 1.1, 1.0)
        @test edge_result == 0.0  # Should be zero outside domain
        
        println("Kernel functions test passed")
    end
    
    println("All KPM tests completed successfully!")
    =#
    
end

=#
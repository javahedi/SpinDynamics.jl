

using Test
using SpinDynamics
using LinearAlgebra



@testset "Basis: full Hilbert space" begin
    L = 4
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
    L = 6
    nup = 3
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
    L = 4
    nup = 2

    # Example parameters
    hopping = [(0,1,1.0), (1,2,0.5)]
    onsite = [0.1, -0.2, 0.3, 0.0]
    zz = [(0,1,0.4), (2,3,-0.6)]
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
const L = 4                     # small system for quick tests
const nup = 2
const t = 0.1
const hopping = [(i,i+1,2.0) for i in 1:(L-1)]
const zz = [(i,i+1,0.0) for i in 1:(L-1)]
const onsite = zeros(L)
const tol = 1e-8

# -----------------------------
# Helper function to compare S(q)
# -----------------------------
function compare_Sq(Sq1::Dict, Sq2::Dict; tol=1e-8)
    keys1 = sort(collect(keys(Sq1)))
    keys2 = sort(collect(keys(Sq2)))
    @assert keys1 == keys2 "Momentum grids differ"
    return all(abs.(values(Sq1) .- values(Sq2)) .< tol)
end

# -----------------------------
# Krylov tests
# -----------------------------
@testset "Krylov module" begin

    # -----------------------------
    # 1) Sector vs full evolution
    # -----------------------------
    mags_sector, Sq_sector = run_krylov_sector(L, nup, hopping, onsite, zz, t)
    mags_full,   Sq_full   = run_krylov_full(L, hopping, onsite, zz, t)

    # The first nup entries of full should match sector (domain-wall initial state)
    #@show mags_full  
    #@show mags_sector
    
    @test all(abs.(mags_sector .- mags_full) .< tol)
    @test compare_Sq(Sq_sector, Sq_full; tol=tol)

    # -----------------------------
    # 2) Krylov time evolution preserves norm
    # -----------------------------
    states, idxmap = build_sector_basis(L, nup)
    ψ0 = domain_wall_state_sector(L, nup, states, idxmap)
    ψ0c = ComplexF64.(ψ0)
    ψt = krylov_time_evolve(ψ0c, t, apply_H_sector!, 
                            SpinParams(L,hopping,onsite,zz),  m=30, states=states, idxmap=idxmap)
    @show norm(ψt)
    @show norm(ψ0c)
    @test abs(norm(ψt) - norm(ψ0c)) < tol
    

    # -----------------------------
    # 3) Krylov evolution reduces to identity at t=0
    # -----------------------------
    ψt0 = krylov_time_evolve(ψ0c, 0.0, apply_H_sector!, 
                            SpinParams(L,hopping,onsite,zz), m=30, states=states, idxmap=idxmap)
    @test all(abs.(ψt0 .- ψ0c) .< tol)

end



# -----------------------------
# Chebyshev tests
# -----------------------------
@testset "Chebyshev norm preservation" begin



   # ----------------------------- # 1) Sector vs full evolution # ----------------------------- 
   mags_sector, Sq_sector, E_bounds= run_chebyshev_sector(L, nup, hopping, onsite, zz, t) 
   mags_full, Sq_full, E_bounds = run_chebyshev_full(L, hopping, onsite, zz, t) 
   @test all(abs.(mags_sector .- mags_full) .< tol) 
   @test compare_Sq(Sq_sector, Sq_full; tol=tol)



    
    
    states, idxmap = build_sector_basis(L, nup)
    ψ0 = domain_wall_state_sector(L, nup, states, idxmap)
    ψ0c = ComplexF64.(ψ0)
    ψt_sector = chebyshev_time_evolve(ψ0c, t, apply_H_sector!, 
                                      SpinParams(L, hopping, onsite, zz),
                                      M=500, E_bounds=E_bounds,
                                      states=states, idxmap=idxmap)
    @show norm(ψt_sector)
    @show norm(ψ0c)
    @test abs(norm(ψt_sector) - norm(ψ0c)) < tol

    

    # Full Hilbert space evolution

    ψ0_full = zeros(ComplexF64, 1<<L)
    state_dw = domain_wall_state_full(L)
    ψ0_full[Int(state_dw)+1] = 1.0 + 0im
    ψ0c_full = copy(ψ0_full)
    ψt_full = chebyshev_time_evolve(ψ0c_full, t, apply_H_full!, 
                                    SpinParams(L, hopping, onsite, zz),
                                    M=500, E_bounds=E_bounds)
    @show norm(ψt_full)
    @show norm(ψ0c_full)
    @test abs(norm(ψt_full) - norm(ψ0c_full)) < tol
    
end

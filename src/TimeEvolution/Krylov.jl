module Krylov



"""
Krylov time-evolution submodule for SpinDynamics.
Provides Krylov/Lanczos time evolution for both:
 - sector-restricted real vectors (Float64)
 - full Hilbert complex vectors (ComplexF64)
"""

using LinearAlgebra
# import sibling modules from parent (SpinDynamics)
using ...Basis
using ...Hamiltonian
using ...InitialStates
using ...Observables

export krylov_time_evolve
export run_krylov_sector, run_krylov_full

# ---------------------------------------------------------
# Krylov time evolution 
# ---------------------------------------------------------
"""
krylov_time_evolve(ψ0, t, applyH!, p, states, idxmap; m=30)

- ψ0: Vector{ComplexF64} (sector basis)
- applyH!: function of signature (out, vec, p, states, idxmap)
- p: SpinParams
- states, idxmap: sector basis info
"""
function krylov_time_evolve(ψ0::AbstractVector{T}, dt::Float64,
        applyH!, p::SpinParams ; m::Int=30, 
        states=nothing, idxmap=nothing) where T<:Number

    n = length(ψ0)
    V = Vector{Vector{T}}(undef, m)
    α = zeros(Float64, m)
    β = zeros(Float64, m-1)
    w = zeros(T, n)

    norm0 = norm(ψ0)
    if norm0 == 0
        return copy(ψ0)
    end
    V[1] = copy(ψ0) / norm0

    m_eff = m
    for j in 1:m

        if states === nothing
            applyH!(w, V[j], p)
        else
            applyH!(w, V[j], p, states, idxmap)
        end

        
        α[j] = dot(V[j], w)
        w .-= α[j] .* V[j]
        if j > 1
            w .-= β[j-1] .* V[j-1]
        end
        if j < m
            β[j] = norm(w)
            if β[j] < 1e-14
                m_eff = j
                α = α[1:m_eff]
                β = β[1:(m_eff-1)]
                V = V[1:m_eff]
                break
            end
            V[j+1] = copy(w / β[j])
        end
    end

    # build tridiagonal and exponentiate
    
    TR = SymTridiagonal(α[1:m_eff], β[1:(m_eff-1)])

    eig = eigen(Hermitian(TR))

    D = eig.values
    Q = eig.vectors
    U_T = Q * Diagonal(exp.(-1im .* D .* dt)) * Q'
    e1 = zeros(T, m_eff); e1[1] = norm0
    y = U_T * e1

    # reconstruct complex ψt (time evolution produces complex amplitudes)

    ψt = zeros(T, n)
    for k in 1:m_eff
        ψt .+= y[k] .* V[k]
    end

    # Renormalize for exact norm
    ψt ./= norm(ψt)

    return ψt
end



# ---------------------------------------------------------
# High-level wrapper: Krylov (sector)
# ---------------------------------------------------------
"""
run_krylov_sector(L, nup, hopping, h, zz, t; m=30)

Builds sector basis, domain-wall initial state, runs Krylov evolution and returns observables.
"""
function run_krylov_sector(L::Int, nup::Int, hopping, h, zz, dt::Float64; m::Int=30)

    p = SpinParams(L,hopping,h,zz)
    states, idxmap = build_sector_basis(L,nup)
    ψ0 = domain_wall_state_sector(L,nup,states,idxmap)

    # Time evolution
    ψt = krylov_time_evolve(ComplexF64.(ψ0), dt, apply_H_sector!, 
                            p, m=m,  states=states, idxmap=idxmap)

    # Observables
    mags = magnetization_per_site_sector(ψt, p, states)
    Sq   = structure_factor_Sq_sector(ψt, p, states)
    return mags, Sq
end

# ---------------------------------------------------------
# High-level wrapper: Krylov (full Hilbert space)
# ---------------------------------------------------------
function run_krylov_full(L::Int, hopping, h, zz, dt::Float64; m::Int=30)

    p = SpinParams(L,hopping,h,zz)
    N = 1<<L
    #ψ0 = zeros(ComplexF64, N)
    #ψ0[1] = 1.0 + 0im  # dN = 2^L, so idx = 1 corresponds to |000…0⟩ (all spins down) in your binary-to-UInt64 mapping
    state_dw = domain_wall_state_full(L)
    idx_dw = Int(state_dw) + 1   # because you index states 0-based -> 1-based Julia array
    ψ0 = zeros(ComplexF64, N)
    ψ0[idx_dw] = 1.0 + 0im


    # Time evolution
    ψt = krylov_time_evolve(ψ0, dt, apply_H_full!, p, m=m)

    # Observables
    mags = magnetization_per_site(ψt, p)
    Sq   = structure_factor_Sq(ψt, p)
    return mags, Sq
end

end # module

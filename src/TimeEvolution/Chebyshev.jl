module Chebyshev

using LinearAlgebra
using SpecialFunctions
using FFTW
using ...Basis
using ...Hamiltonian
using ...InitialStates
using ...Observables

export chebyshev_time_evolve, lanczos_extremal
export  run_chebyshev_sector, run_chebyshev_full



# Debugging utility
function check_vector_consistency(name, vec, expected_norm_range=(0.9, 1.1))
    norm_val = norm(vec)
    has_nan = any(isnan, vec)
    has_inf = any(isinf, vec)
    
    if has_nan || has_inf || norm_val < expected_norm_range[1] || norm_val > expected_norm_range[2]
        println("DEBUG $name: norm=$norm_val, NaN=$has_nan, Inf=$has_inf")
        return false
    end
    return true
end



# ---------------------------------------------------------
# Chebyshev time evolution (full Hilbert space)
function chebyshev_time_evolve(ψ0::Vector{<:Number}, t::Float64,
                               applyH!, p::SpinParams;
                               M::Int=300, E_bounds::Tuple{Float64,Float64}=(-1.0,1.0),
                               states=nothing, idxmap=nothing)

    N = length(ψ0)
    ψt = zeros(ComplexF64, N)
    Emin, Emax = E_bounds
    a = (Emax - Emin)/2
    b = (Emax + Emin)/2

    φ0 = ComplexF64.(ψ0)
    φ1 = zeros(ComplexF64, N)
    φtmp = zeros(ComplexF64, N)

    # First step
    if states === nothing
        applyH!(φ1, φ0, p)
    else
        applyH!(φ1, φ0, p, states, idxmap)
    end
    @. φ1 = (φ1 - b*φ0)/a

    @. ψt = besselj(0, a*t) * φ0 + 2im * besselj(1, a*t) * φ1

    for k in 2:M
        fill!(φtmp, 0.0)
        if states === nothing
            applyH!(φtmp, φ1, p)
        else
            applyH!(φtmp, φ1, p, states, idxmap)
        end
        @. φtmp = (φtmp - b*φ1)/a
        @. φtmp = 2*φtmp - φ0
        ck = 2.0 * im^k * besselj(k, a*t)
        @. ψt += ck * φtmp
        φ0, φ1 = φ1, φtmp
    end

    # Renormalize for exact norm
    ψt ./= norm(ψt)

    return ψt
end





# ---------------------------------------------------------
# Lanczos extremal eigenvalues (sector)
# ---------------------------------------------------------
function lanczos_extremal(applyH!, ψ0::AbstractVector{T}, p::SpinParams;
            m::Int=80, states=nothing, idxmap=nothing) where T<:Number

    N = length(ψ0)
    V = zeros(T, N, m)
    α = zeros(Float64, m)
    β = zeros(Float64, m-1)
    
    # Normalize initial vector
    v = copy(ψ0)
    v_norm = norm(v)
    if v_norm < 1e-12
        error("Initial vector has zero norm")
    end
    v ./= v_norm
    V[:,1] = v
    
    w = zeros(T, N)

    for j in 1:m
        # Apply Hamiltonian
        if states === nothing
            applyH!(w, V[:,j], p)
        else
            applyH!(w, V[:,j], p, states, idxmap)
        end
      
        
        # Orthogonalize against previous vectors
        for k in 1:j
            α[j] = real(dot(V[:,k], w))  # Ensure real value for symmetric H
            @. w -= α[j] * V[:,k]
        end
        
        if j < m
            β[j] = norm(w)
            if β[j] < 1e-12
                # Early termination if breakdown occurs
                α = α[1:j]
                β = β[1:j-1]
                break
            end
            V[:,j+1] = w / β[j]
        end
    end

    # Handle case where we broke down early
    actual_m = length(α)
    if actual_m < m
        β = β[1:actual_m-1]
    end
    
    TR = SymTridiagonal(α, β)
    evals = eigen(TR).values
    return minimum(evals), maximum(evals)
end




# ---------------------------------------------------------
# High-level wrapper: Chebyshev (sector)
# ---------------------------------------------------------
function run_chebyshev_sector(L::Int, nup::Int, hopping, h, 
                                zz, t::Float64; M::Int=500, lanc_m::Int=80)

    p = SpinParams(L, hopping, h, zz)
    states, idxmap = build_sector_basis(L, nup)
    ψ0 = domain_wall_state_sector(L, nup, states, idxmap)

    # Convert to complex
    ψ0 = ComplexF64.(ψ0)
    ψ0_lanczos = copy(ψ0)

    # Estimate Emax
    _, Emax = lanczos_extremal(apply_H_sector!, ψ0_lanczos, p, 
                            m=lanc_m, states=states, idxmap= idxmap)

    # Estimate Emin using -H
    function apply_H_neg!(out, ψ, p, states, idxmap)
        apply_H_sector!(out, ψ, p, states, idxmap)
        @. out = -out
    end
    _, Emax_neg = lanczos_extremal(apply_H_neg!, ψ0_lanczos, p,  
                                    m=lanc_m, states=states, idxmap= idxmap)
    Emin = -Emax_neg

    #@show "sector" , Emin, Emax
    E_bounds = (Emin - 1e-6, Emax + 1e-6)

    # Time evolution
    ψt = chebyshev_time_evolve(ψ0, t, apply_H_sector!, p,  
                                M=M, E_bounds=E_bounds, states=states, idxmap=idxmap)

                                  

    # Observables
    mags = magnetization_per_site_sector(ψt, p, states)
    Sq = structure_factor_Sq_sector(ψt, p, states)
    
    return mags, Sq, E_bounds
end

# ---------------------------------------------------------
# High-level wrapper: Chebyshev (full Hilbert space)
# ---------------------------------------------------------
function run_chebyshev_full(L::Int, hopping, h, zz, t::Float64; M::Int=500, lanc_m::Int=80)

    p = SpinParams(L, hopping, h, zz)
    
    # Create domain wall state
    N = 1 << L  # Hilbert space dimension
   # Convert state representation to vector
    state_dw = domain_wall_state_full(L)
    ψ0 = zeros(ComplexF64, N)
    ψ0[Int(state_dw) + 1] = 1.0 + 0im  # Convert from 0-based to 1-based indexing


    ψ0_lanczos = copy(ψ0)

    # Estimate Emax
    _, Emax = lanczos_extremal(apply_H_full!, ψ0_lanczos, p, m=lanc_m)

    # Estimate Emin using -H
    function apply_H_neg!(out, ψ, p)
        apply_H_full!(out, ψ, p)
        @. out = -out
    end
    _, Emax_neg = lanczos_extremal(apply_H_neg!, ψ0_lanczos, p, m=lanc_m)
    Emin = -Emax_neg

    #@show "full" , Emin, Emax
    E_bounds = (Emin - 1e-6, Emax + 1e-6)

    # Time evolution
    ψt = chebyshev_time_evolve(ψ0, t, apply_H_full!, p, M=M, E_bounds=E_bounds)

    # Observables
    mags = magnetization_per_site(ψt, p)
    Sq = structure_factor_Sq(ψt, p)
    
    return mags, Sq, E_bounds
end

end # module


"""

#=
# ---------------------------------------------------------
# Lanczos extremal eigenvalues (full Hilbert space)
# ---------------------------------------------------------
function lanczos_extremal_full(applyH!, ψ0::AbstractVector{T}, 
                                p::SpinParams; m::Int=100) where T<:Number

    N = length(ψ0)
    V = zeros(T, N, m)
    α = zeros(Float64, m)
    β = zeros(Float64, m-1)
    
    # Normalize initial vector
    v = copy(ψ0)
    v_norm = norm(v)
    if v_norm < 1e-12
        error("Initial vector has zero norm")
    end
    v ./= v_norm
    V[:,1] = v
    
    w = zeros(T, N)

    for j in 1:m
        # Apply Hamiltonian
        applyH!(w, V[:,j], p)
        
        # Orthogonalize against previous vectors
        for k in 1:j
            α[j] = real(dot(V[:,k], w))  # Ensure real value for symmetric H
            @. w -= α[j] * V[:,k]
        end
        
        if j < m
            β[j] = norm(w)
            if β[j] < 1e-12
                # Early termination if breakdown occurs
                α = α[1:j]
                β = β[1:j-1]
                break
            end
            V[:,j+1] = w / β[j]
        end
    end

    # Handle case where we broke down early
    actual_m = length(α)
    if actual_m < m
        β = β[1:actual_m-1]
    end
    
    TR = SymTridiagonal(α, β)
    evals = eigen(TR).values
    return minimum(evals), maximum(evals)
end
#=

"""
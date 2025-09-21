module Chebyshev

using LinearAlgebra
using SpecialFunctions: besselj
using ArnoldiMethod
using FFTW

# Import from parent module's submodules
using ...Basis
using ...Hamiltonian
using ...InitialStates
using ...Observables

export chebyshev_time_evolve, lanczos_extremal, estimate_energy_bounds
export run_chebyshev_sector, run_chebyshev_full, ChebyshevWorkspace

# Pre-allocate workspace for Chebyshev evolution to avoid repeated allocations
mutable struct ChebyshevWorkspace{T<:Number}
    ϕ_prev::Vector{T}
    ϕ_curr::Vector{T}
    ϕ_next::Vector{T}
    ψ_t::Vector{T}
    w::Vector{T}  # workspace for Hamiltonian application
    
    function ChebyshevWorkspace{T}(N::Int) where T<:Number
        ϕ_prev = Vector{T}(undef, N)
        ϕ_curr = Vector{T}(undef, N)
        ϕ_next = Vector{T}(undef, N)
        ψ_t = Vector{T}(undef, N)
        w = Vector{T}(undef, N)
        new(ϕ_prev, ϕ_curr, ϕ_next, ψ_t, w)
    end
end

ChebyshevWorkspace(ψ::AbstractVector{T}) where T<:Number = ChebyshevWorkspace{T}(length(ψ))

"""
    apply_rescaled_H!(out, ψ, applyH!, p, states, idxmap, a, b)

Compute out .= (Hψ - b*ψ) / a, where Hψ is written by applyH!(out, ψ, p[, states, idxmap]).
- `out` and `ψ` must be preallocated and have the same length.
- This function performs all operations in-place and avoids temporaries.
Returns `out`.
"""
function apply_rescaled_H!(out::AbstractVector{T}, ψ::AbstractVector{T},
                        applyH!, p, states, idxmap, a::Float64, b::Float64) where T<:Number
    @assert length(out) == length(ψ)
    
    # Fill out with H * ψ via the user-provided applyH!
    if states === nothing
        applyH!(out, ψ, p)
    else
        applyH!(out, ψ, p, states, idxmap)
    end
    
    # In-place rescaling: out = (out - b * ψ) / a
    @inbounds for i in eachindex(out)
        out[i] = (out[i] - b * ψ[i]) / a
    end
    
    return out
end

"""
    chebyshev_time_evolve(ψ0, dt, applyH!, p; n=100, Ebounds=(-1.0,1.0), 
                          states=nothing, idxmap=nothing, workspace=nothing)

Chebyshev time evolution for one time-step `dt`, using `n` Chebyshev terms.

Arguments:
- ψ0 : initial state (vector)
- dt : time step (Float64)
- applyH! : function to compute Hψ, signature either
    `applyH!(out, ψ, p)` or `applyH!(out, ψ, p, states, idxmap)` and writes into `out`
- p : user params (e.g., SpinParams)

Keyword args:
- n : number of Chebyshev terms (default 100)
- Ebounds : tuple (E_min, E_max) for rescaling H
- states, idxmap : optional extra arguments forwarded to applyH!
- workspace : optional preallocated ChebyshevWorkspace to avoid allocations

Returns:
- ψ_t (complex-promoted vector) containing the time-evolved state.
"""
function chebyshev_time_evolve(ψ0::AbstractVector{T}, dt::Float64,
                            applyH!, p; n::Int=100,
                            Ebounds::Tuple{Float64,Float64}=(-1.0,1.0),
                            states=nothing, idxmap=nothing, 
                            workspace=nothing) where T<:Number
    @assert n >= 1 "n must be >= 1"
    N = length(ψ0)
    E_min, E_max = Ebounds
    
    # Rescaling constants
    a = (E_max - E_min) / (2 * 0.9999)  # slight shrink to avoid edge issues
    b = (E_max + E_min) / 2
    
    # Precompute Chebyshev coefficients (complex)
    c = Vector{ComplexF64}(undef, n)
    phase_factor = exp(-im * b * dt)
    for k in 0:n-1
        delta_k0 = (k == 0) ? 1.0 : 0.0
        c[k+1] = (2 - delta_k0) * (-im)^k * besselj(k, a * dt) * phase_factor
    end
    
    # Prepare workspace
    if workspace === nothing
        ws = ChebyshevWorkspace(ψ0)
    else
        ws = workspace
        @assert length(ws.ϕ_prev) == N "Workspace size mismatch"
    end
    
    # Initialize recurrence
    copyto!(ws.ϕ_prev, ψ0)  # T₀(Ḣ)ψ0 = ψ0
    
    # Compute T1(Ḣ)ψ0 -> ϕ_curr = Ḣ * ϕ_prev
    apply_rescaled_H!(ws.ϕ_curr, ws.ϕ_prev, applyH!, p, states, idxmap, a, b)
    
    # Start ψ_t = c0 * T0 + c1 * T1
    fill!(ws.ψ_t, zero(eltype(ws.ψ_t)))
    @inbounds for i in eachindex(ws.ψ_t)
        ws.ψ_t[i] += c[1] * ws.ϕ_prev[i]
        if n >= 2
            ws.ψ_t[i] += c[2] * ws.ϕ_curr[i]
        end
    end
    
    # If n == 1, we're done
    if n == 1
        return copy(ws.ψ_t)
    end
    
    # Recurrence for k = 2 .. n-1
    for k in 2:n-1
        # ϕ_next := 2 * Ḣ * ϕ_curr - ϕ_prev
        apply_rescaled_H!(ws.ϕ_next, ws.ϕ_curr, applyH!, p, states, idxmap, a, b)
        
        @inbounds for i in eachindex(ws.ϕ_next)
            ws.ϕ_next[i] = 2 * ws.ϕ_next[i] - ws.ϕ_prev[i]
            ws.ψ_t[i] += c[k+1] * ws.ϕ_next[i]
        end
        
        # Rotate buffers
        ws.ϕ_prev, ws.ϕ_curr, ws.ϕ_next = ws.ϕ_curr, ws.ϕ_next, ws.ϕ_prev
    end
    
    return copy(ws.ψ_t)
end

"""
    lanczos_extremal(applyH!, ψ0, p; m=80, states=nothing, idxmap=nothing, tol=1e-12)

Compute extremal eigenvalues using Lanczos algorithm.

Arguments:
- applyH! : Hamiltonian application function
- ψ0 : initial vector
- p : SpinParams
- m : number of Lanczos iterations
- states, idxmap : optional for sector representation
- tol : tolerance for breakdown detection

Returns:
- (Emin, Emax) : minimum and maximum eigenvalues
"""
function lanczos_extremal(applyH!, ψ0::AbstractVector{T}, p;
                          m::Int=80, states=nothing, idxmap=nothing, 
                          tol::Float64=1e-12) where T<:Number
    N = length(ψ0)
    α = zeros(Float64, m)
    β = zeros(Float64, m-1)
    
    # Normalize initial vector
    v_prev = copy(ψ0)
    v_prev_norm = norm(v_prev)
    if v_prev_norm < tol
        error("Initial vector has zero norm")
    end
    v_prev ./= v_prev_norm
    
    w = similar(ψ0)
    v_curr = similar(ψ0)
    
    for j in 1:m
        # Apply Hamiltonian
        if states === nothing
            applyH!(w, v_prev, p)
        else
            applyH!(w, v_prev, p, states, idxmap)
        end
        
        # Compute α[j] = ⟨v_prev|H|v_prev⟩
        α[j] = real(dot(v_prev, w))
        
        # Orthogonalize against previous vectors
        if j == 1
            @. w -= α[j] * v_prev
        else
            @. w -= α[j] * v_prev + β[j-1] * v_curr
        end
        
        if j < m
            β[j] = norm(w)
            if β[j] < tol
                # Early termination if breakdown occurs
                α = α[1:j]
                β = β[1:j-1]
                break
            end
            
            # Prepare for next iteration
            v_curr, v_prev = v_prev, w / β[j]  # Swap and normalize
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

"""
    estimate_energy_bounds(applyH!, ψ0, p; m=80, states=nothing, idxmap=nothing)

Estimate the minimum and maximum eigenvalues (Emin, Emax) of the Hamiltonian 
using Lanczos applied to `ψ0`.
"""
function estimate_energy_bounds(applyH!, ψ0::AbstractVector, p;
                                m::Int=80, states=nothing, idxmap=nothing)
    # Estimate Emax
    _, Emax = lanczos_extremal(applyH!, ψ0, p; m=m, states=states, idxmap=idxmap)
    
    # Define negative Hamiltonian
    function apply_H_neg!(out, ψ, p, states=nothing, idxmap=nothing)
        if states === nothing
            applyH!(out, ψ, p)
        else
            applyH!(out, ψ, p, states, idxmap)
        end
        @. out = -out
    end
    
    _, Emax_neg = lanczos_extremal(apply_H_neg!, ψ0, p; m=m, states=states, idxmap=idxmap)
    Emin = -Emax_neg
    
    return Emin, Emax
end

"""
    run_chebyshev_sector(L, nup, hopping, h, zz, dt; n=50, lanc_m=80)

High-level wrapper for Chebyshev time evolution in a magnetization sector.
"""
function run_chebyshev_sector(L::Int, nup::Int, hopping, h, zz, dt::Float64; 
                              n::Int=50, lanc_m::Int=80)
    p = SpinParams(L, hopping, h, zz)
    states, idxmap = build_sector_basis(L, nup)
    ψ0 = domain_wall_state_sector(L, nup, states, idxmap)
    
    # Convert to complex and normalize
    ψ0 = ComplexF64.(ψ0)
    ψ0 ./= norm(ψ0)
    
    Emin, Emax = estimate_energy_bounds(apply_H_sector!, ψ0, p; 
                                    m=lanc_m, states=states, idxmap=idxmap)
    
    # Time evolution
    ψt = chebyshev_time_evolve(ψ0, dt, apply_H_sector!, p, Ebounds=(Emin, Emax), 
                                n=n, states=states, idxmap=idxmap)
    
    # Observables
    mags = magnetization_per_site_sector(ψt, p, states)
    Sq = structure_factor_Sq_sector(ψt, p, states)
    
    return mags, Sq, (Emin, Emax)
end

"""
    run_chebyshev_full(L, hopping, h, zz, dt; n=500, lanc_m=80)

High-level wrapper for Chebyshev time evolution in full Hilbert space.
"""
function run_chebyshev_full(L::Int, hopping, h, zz, dt::Float64; 
                            n::Int=500, lanc_m::Int=80)
    p = SpinParams(L, hopping, h, zz)
    
    # Create domain wall state
    N = 1 << L
    state_dw = domain_wall_state_full(L)
    ψ0 = zeros(ComplexF64, N)
    ψ0[Int(state_dw) + 1] = 1.0
    
    # Normalize
    ψ0 ./= norm(ψ0)
    
    Emin, Emax = estimate_energy_bounds(apply_H_full!, ψ0, p; m=lanc_m)
    
    # Time evolution
    ψt = chebyshev_time_evolve(ψ0, dt, apply_H_full!, p, n=n, Ebounds=(Emin, Emax))
    
    # Observables
    mags = magnetization_per_site(ψt, p)
    Sq = structure_factor_Sq(ψt, p)
    
    return mags, Sq, (Emin, Emax)
end

end # module Chebyshev
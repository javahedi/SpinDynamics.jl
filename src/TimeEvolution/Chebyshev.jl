module Chebyshev

    using LinearAlgebra
    using SpecialFunctions: besselj
    using ArnoldiMethod
    using FFTW

    # Import from parent module's submodules
    using ...SpinModel
    using ...Basis
    using ...Hamiltonian
    using ...InitialStates
    using ...Observables

    export chebyshev_time_evolve, lanczos_extremal, estimate_energy_bounds, 
            run_chebyshev, ChebyshevWorkspace, apply_rescaled_H!

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
        apply_rescaled_H!(out, ψ, applyH!, model, idxmap, a, b)

    Compute out .= (Hψ - b*ψ) / a, where Hψ is written by applyH!(out, ψ, p[, states, idxmap]).
    - `out` and `ψ` must be preallocated and have the same length.
    - This function performs all operations in-place and avoids temporaries.
    Returns `out`.
    """
    function apply_rescaled_H!(out::AbstractVector{T}, ψ::AbstractVector{T},
                            applyH!, model::SpinModel.Model, a::Float64, b::Float64) where T<:Number
        @assert length(out) == length(ψ)
        
        # Fill out with H * ψ via the user-provided applyH!
        applyH!(out, ψ, model)
        
        
        # In-place rescaling: out = (out - b * ψ) / a
        @inbounds for i in eachindex(out)
            out[i] = (out[i] - b * ψ[i]) / a
        end
        
        return out
    end

    """
        chebyshev_time_evolve(ψ0, dt, applyH!, model; n=100, Ebounds=(-1.0,1.0), 
                            workspace=nothing)

    Chebyshev time evolution for one time-step `dt`, using `n` Chebyshev terms.

    Arguments:
    - ψ0 : initial state (vector)
    - dt : time step (Float64)
    - applyH! : function to compute Hψ, signature either
        `applyH!(out, ψ, model)` 
    - model : SpinModel.Model instance

    Keyword args:
    - n : number of Chebyshev terms (default 100)
    - Ebounds : tuple (E_min, E_max) for rescaling H
    - states, idxmap : optional extra arguments forwarded to applyH!
    - workspace : optional preallocated ChebyshevWorkspace to avoid allocations

    Returns:
    - ψ_t (complex-promoted vector) containing the time-evolved state.
    """
    function chebyshev_time_evolve(ψ0::AbstractVector{T}, dt::Float64,
                                applyH!, model::SpinModel.Model; cheb_n::Int=100,
                                Ebounds::Tuple{Float64,Float64}=(-1.0,1.0),
                                workspace=nothing) where T<:Number
        @assert cheb_n >= 1 "cheb_n must be >= 1"
        N = length(ψ0)
        E_min, E_max = Ebounds
        
        # Rescaling constants
        a = (E_max - E_min) / (2 * 0.9999)  # slight shrink to avoid edge issues
        b = (E_max + E_min) / 2
        
        # Precompute Chebyshev coefficients (complex)
        c = Vector{ComplexF64}(undef, cheb_n)
        phase_factor = exp(-im * b * dt)
        for k in 0:cheb_n-1
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
        apply_rescaled_H!(ws.ϕ_curr, ws.ϕ_prev, applyH!, model, a, b)
        
        # Start ψ_t = c0 * T0 + c1 * T1
        fill!(ws.ψ_t, zero(eltype(ws.ψ_t)))
        @inbounds for i in eachindex(ws.ψ_t)
            ws.ψ_t[i] += c[1] * ws.ϕ_prev[i]
            if cheb_n >= 2
                ws.ψ_t[i] += c[2] * ws.ϕ_curr[i]
            end
        end
        
        # If n == 1, we're done
        if cheb_n == 1
            return copy(ws.ψ_t)
        end
        
        # Recurrence for k = 2 .. n-1
        for k in 2:cheb_n-1
            # ϕ_next := 2 * Ḣ * ϕ_curr - ϕ_prev
            apply_rescaled_H!(ws.ϕ_next, ws.ϕ_curr, applyH!, model, a, b)
            
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
        lanczos_extremal(applyH!,  model; m=80, tol=1e-12)

    Compute extremal eigenvalues using Lanczos algorithm.

    Arguments:
    - applyH! : Hamiltonian application function
    - model :   SpinModel.Model instance
    - m : number of Lanczos iterations

    - tol : tolerance for breakdown detection

    Returns:
    - (Emin, Emax) : minimum and maximum eigenvalues
    """
    function lanczos_extremal(applyH!, N::Int, model::SpinModel.Model; 
                            lanc_m::Int=80, 
                            tol::Float64=1e-12) 
        # N = dimension of the Hilbert space
        # Initialize random starting vector
        ψ0 = randn(ComplexF64, N)  # complex random vector
        ψ0 ./= norm(ψ0)

        α = zeros(Float64, lanc_m)
        β = zeros(Float64, lanc_m-1)

        v_prev = copy(ψ0)
        w = similar(ψ0)
        v_curr = similar(ψ0)

        for j in 1:lanc_m
            # Apply Hamiltonian
            applyH!(w, v_prev, model)
         

            # α[j] = <v_prev|H|v_prev>
            α[j] = real(dot(v_prev, w))

            # Orthogonalize
            if j == 1
                @. w -= α[j] * v_prev
            else
                @. w -= α[j] * v_prev + β[j-1] * v_curr
            end

            if j < lanc_m
                β[j] = norm(w)
                if β[j] < tol
                    α = α[1:j]
                    β = β[1:j-1]
                    break
                end
                v_curr, v_prev = v_prev, w / β[j]
            end
        end

        actual_m = length(α)
        if actual_m < lanc_m
            β = β[1:actual_m-1]
        end

        TR = SymTridiagonal(α, β)
        evals = eigen(TR).values
        return minimum(evals), maximum(evals)
    end


    """
        estimate_energy_bounds(applyH!, N::Int, model::SpinModel.Model; m=80)

    Estimate the minimum and maximum eigenvalues (Emin, Emax) of the Hamiltonian 
    using Lanczos applied to `ψ0`.
    """
    function estimate_energy_bounds(applyH!, N::Int, model::SpinModel.Model;
                                    lanc_m::Int=80)
        # Estimate Emax
        _, Emax = lanczos_extremal(applyH!, N, model; lanc_m=lanc_m)

        # Define negative Hamiltonian
        function apply_H_neg!(out, ψ, model)
                applyH!(out, ψ, model)
            @. out = -out
        end

        _, Emax_neg = lanczos_extremal(apply_H_neg!, N, model; lanc_m=lanc_m)
        Emin = -Emax_neg
        
        return Emin, Emax
    end


    """
        run_chebyshev(L, nup, hopping, h, zz, dt; n=50, lanc_m=80)

    High-level wrapper for Chebyshev time evolution in a magnetization sector.
    """
    function run_chebyshev(model::SpinModel.Model, dt::Float64; 
                                cheb_n::Int=50, lanc_m::Int=80)
        

        ψ0 = domain_wall_state(model)
        # Convert to complex and normalize
        ψ0 = ComplexF64.(ψ0)
        ψ0 ./= norm(ψ0)

        Emin, Emax = estimate_energy_bounds(apply_H!, length(ψ0), model; 
                                        lanc_m=lanc_m)

        # Time evolution
        ψt = chebyshev_time_evolve(ψ0, dt, apply_H!, model, Ebounds=(Emin, Emax), 
                                    cheb_n=cheb_n)
        
        # Observables
        mags = magnetization_per_site(ψt, model)
        Sq = structure_factor_Sq(ψt, model)

        return mags, Sq, (Emin, Emax)
    end

       

end # module Chebyshev
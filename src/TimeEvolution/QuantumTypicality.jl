module QuantumTypicality

    using LinearAlgebra
    using ...Basis
    using ...Hamiltonian
    using ...InitialStates
    using ...Observables
    using ..Krylov # Import the sibling Krylov module to use its time evolution function
    using ..Chebyshev # Import the sibling Chebyshev module to use its time evolution function

    export typicality_correlation_function
    export rk4_time_step

    """
        typicality_correlation_function(L, β, operator_i, operator_j, t_range; method=:krylov, kwargs...)
    Calculates the finite-temperature correlation function <A(t)B(0)>_β using quantum typicality.
    - `L`: System size
    - `β`: Inverse temperature
    - `operator_i`: Function (ψ, p, states, idxmap) -> A|ψ> (applies operator A, e.g., s_z on a site)
    - `operator_j`: Function (ψ, p, states, idxmap) -> B|ψ> (applies operator B)
    - `t_range`: Time points to evaluate
    - `method`: Time evolution method (:krylov, :chebyshev, :rk4)
    - `kwargs`: Additional arguments for the time evolution methods (e.g., `m` for Krylov, `n` for Chebyshev, `dt` for RK4)
    Returns: Vector of complex correlation values for each time in `t_range`.

    NOTE: At temperature T=0 (β→∞), <A(t)B(0)>_β =<GS|A(t)B(0)|GS> reduces to the ground state correlation function.
        This is because the thermal state |ψ_β⟩ approaches the ground state |GS⟩ as β increases.    
        <GS|A(t)B(0)|GS> = <GS|e^{iHt} A e^{-iHt} B|GS> = <GS|A(t)B(0)|GS>
        we can also use KPM method to calulate 
        dynamical correlation S(q,ω)  and spectral function A(q,ω) at T=0.
         
    """
    function typicality_correlation_function(L::Int, β::Float64,
                                            operator_i, operator_j,
                                            t_range::Vector{Float64};
                                            method::Symbol=:krylov,
                                            kwargs...)
        # 1. Build the sector basis (or full basis) - adjust as needed for your system
        # For example, let's assume we're in the zero magnetization sector
        nup = L ÷ 2
        p = SpinParams(L, hopping, h, zz) # You'll need to define these or pass them in
        states, idxmap = build_sector_basis(L, nup)
        
        # 2. Generate a random initial state in the computational basis
        dim = length(states)
        r_vec = randn(ComplexF64, dim)
        r_vec ./= norm(r_vec)
        
        # 3. Prepare the thermal state |ψ_β⟩ = e^{-βH/2} |r⟩
        #    This requires imaginary time evolution - we can use Krylov for this
        println("Preparing thermal state with β = $β...") 
        ψ_β = krylov_imaginary_time_evolution(r_vec, β/2, p, states, idxmap; kwargs...)
        normalization = norm(ψ_β)
        ψ_β ./= normalization
        
        # 4. Apply operator B to get the initial state for real time evolution
        println("Applying operator B...")
        ϕ0 = operator_j(ψ_β, p, states, idxmap)
        
        # 5. Time evolution of |ϕ(t)⟩ = e^{-iHt} |ϕ₀⟩
        println("Starting real time evolution using $method method...")
        
        # Preallocate results
        results = zeros(ComplexF64, length(t_range))
        
        # Choose the time evolution method
        if method == :krylov
            evolve! = krylov_evolver
        elseif method == :chebyshev
            evolve! = chebyshev_evolver
        elseif method == :rk4
            evolve! = rk4_evolver
        else
            error("Unknown method: $method")
        end
        
        # Store current state and evolve through each time point
        ϕ_t = copy(ϕ0)
        for (i, t) in enumerate(t_range)
            if i > 1
                # Evolve from previous time to current time
                Δt = t - t_range[i-1]
                ϕ_t = evolve!(ϕ_t, Δt, p, states, idxmap; kwargs...)
            end
            
            # Measure the correlation: <ψ_β|A|ϕ(t)>
            results[i] = dot(ψ_β, operator_i(ϕ_t, p, states, idxmap))
        end
        
        return results ./ normalization^2
    end

    # -------------------------------------------------------------------
    # Time evolution methods
    # -------------------------------------------------------------------

    function krylov_evolver(ψ::Vector{ComplexF64}, dt::Float64, 
                        p::SpinParams, states, idxmap; m::Int=30, kwargs...)
        return Krylov.krylov_time_evolve(ψ, dt, apply_H_sector!, p, 
                                    m=m, states=states, idxmap=idxmap)
    end

    function chebyshev_evolver(ψ::Vector{ComplexF64}, dt::Float64,
                            p::SpinParams, states, idxmap; n::Int=100, kwargs...)
        return Chebyshev.chebyshev_time_evolve(ψ, dt, apply_H_sector!, p,
                                n=n, states=states, idxmap=idxmap)
    end

    function rk4_evolver(ψ::Vector{ComplexF64}, dt::Float64,
                        p::SpinParams, states, idxmap; kwargs...)
        return rk4_time_step(ψ, dt, apply_H_sector!, p, states, idxmap)
    end

    # -------------------------------------------------------------------
    # RK4 time evolution implementation
    # -------------------------------------------------------------------

    """
    rk4_time_step(ψ, dt, applyH!, p, states, idxmap)
    RK4 time evolution for a single time step dt.
    """
    function rk4_time_step(ψ::Vector{T}, dt::Float64,
                        applyH!, p::SpinParams,
                        states, idxmap) where T<:Number
        
        n = length(ψ)
        
        # RK4 steps
        k1 = zeros(T, n)
        applyH!(k1, ψ, p, states, idxmap)
        k1 .*= -im * dt
        
        k2 = zeros(T, n)
        applyH!(k2, ψ + k1/2, p, states, idxmap)
        k2 .*= -im * dt
        
        k3 = zeros(T, n)
        applyH!(k3, ψ + k2/2, p, states, idxmap)
        k3 .*= -im * dt
        
        k4 = zeros(T, n)
        applyH!(k4, ψ + k3, p, states, idxmap)
        k4 .*= -im * dt
        
        return ψ + (k1 + 2*k2 + 2*k3 + k4)/6
    end

    # -------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------

    δ(i, j) = i == j ? 1 : 0

    function krylov_imaginary_time_evolution(ψ0::Vector{ComplexF64}, β::Float64,
                                            p::SpinParams, states, idxmap; m::Int=30)
        # Imaginary time evolution using Krylov method
        # Similar to real time but with e^{-βH} instead of e^{-iHt}
        n = length(ψ0)
        V = Vector{Vector{ComplexF64}}(undef, m)
        α = zeros(Float64, m)
        β_kry = zeros(Float64, m-1)
        w = zeros(ComplexF64, n)
        
        norm0 = norm(ψ0)
        V[1] = copy(ψ0) / norm0
        
        m_eff = m
        for j in 1:m
            if states === nothing
                apply_H_sector!(w, V[j], p)
            else
                apply_H_sector!(w, V[j], p, states, idxmap)
            end
            
            α[j] = real(dot(V[j], w))  # For imaginary time, we work with real algebra
            w .-= α[j] .* V[j]
            
            if j > 1
                w .-= β_kry[j-1] .* V[j-1]
            end
            
            if j < m
                β_kry[j] = norm(w)
                if β_kry[j] < 1e-14
                    m_eff = j
                    break
                end
                V[j+1] = copy(w / β_kry[j])
            end
        end
        
        # Exponentiate for imaginary time
        TR = SymTridiagonal(α[1:m_eff], β_kry[1:(m_eff-1)])
        eig = eigen(Hermitian(TR))
        
        D = eig.values
        Q = eig.vectors
        U_T = Q * Diagonal(exp.(-β .* D)) * Q'
        
        e1 = zeros(ComplexF64, m_eff)
        e1[1] = norm0
        y = U_T * e1
        
        # Reconstruct the state
        ψ_β = zeros(ComplexF64, n)
        for k in 1:m_eff
            ψ_β .+= y[k] .* V[k]
        end
        
        return ψ_β
    end

end # module QuantumTypicality
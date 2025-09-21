module Observables

    using FFTW
    using Base.Threads
    using ..Hamiltonian: SpinParams, sz_value, bit_at

    export magnetization_per_site, structure_factor_Sq, connected_correlations
    export magnetization_per_site_sector, structure_factor_Sq_sector, connected_correlations_sector

    # --------------------------------------------------
    # Magnetization per site (full Hilbert space)
    # --------------------------------------------------
    function magnetization_per_site(ψ::AbstractVector{T}, p::SpinParams) where {T<:Number}
        L, N = p.L, length(ψ)
        nth = nthreads()
        local_mags = [zeros(Float64, L) for _ in 1:nth]

        Threads.@threads for idx in 1:N
            prob = abs2(ψ[idx])
            if prob != 0.0
                state = UInt64(idx-1)
                tid = threadid()
                loc = local_mags[tid]
                for i in 0:(L-1)
                    loc[i+1] += prob * sz_value(bit_at(state,i))
                end
            end
        end

        mags = zeros(Float64,L)
        for loc in local_mags
            mags .+= loc
        end
        return mags
    end

    # --------------------------------------------------
    # Connected correlations C_r = <S_i S_{i+r}> - <S_i><S_{i+r}>
    # --------------------------------------------------
    function connected_correlations(ψ::AbstractVector{T}, p::SpinParams) where {T<:Number}
        L, N = p.L, length(ψ)
        nth = nthreads()
        local_sz = [zeros(Float64, N, L) for _ in 1:nth]

        Threads.@threads for idx in 1:N
            prob = abs2(ψ[idx])
            if prob != 0.0
                state = UInt64(idx-1)
                tid = threadid()
                loc = local_sz[tid]
                for i in 0:(L-1)
                    loc[idx,i+1] = sz_value(bit_at(state,i)) * prob
                end
            end
        end

        szvals = zeros(Float64, N, L)
        for loc in local_sz
            szvals .+= loc
        end

        S_i = sum(szvals, dims=1)[:]
        C_r = zeros(Float64,L)
        for r in 0:(L-1)
            tmp = 0.0
            for i in 1:L
                j = mod1(i+r,L)
                tmp += sum(szvals[:,i] .* szvals[:,j]) - S_i[i]*S_i[j]
            end
            C_r[r+1] = tmp / L
        end
        return C_r
    end

    # --------------------------------------------------
    # Spin structure factor S(q)
    # --------------------------------------------------
    function structure_factor_Sq(ψ::AbstractVector{T}, p::SpinParams) where {T<:Number}
        C_r = connected_correlations(ψ, p)
        S_q = fft(C_r)
        qlist = [2π*(n-1)/p.L for n in 1:p.L]
        Sq_dict = Dict{Float64,Float64}()
        for n in 1:p.L
            Sq_dict[qlist[n]] = real(S_q[n])
        end
        return Sq_dict
    end

    # --------------------------------------------------
    # Sector versions
    # --------------------------------------------------
    function magnetization_per_site_sector(ψ::AbstractVector{T}, p::SpinParams, 
                                            states::Vector{UInt64}) where {T<:Number}
        L, N = p.L, length(ψ)
        nth = nthreads()
        local_mags = [zeros(Float64, L) for _ in 1:nth]

        Threads.@threads for idx in 1:N
            prob = abs2(ψ[idx])
            if prob != 0.0
                state = states[idx]
                tid = threadid()
                loc = local_mags[tid]
                for i in 0:(L-1)
                    loc[i+1] += prob * sz_value(bit_at(state,i))
                end
            end
        end

        mags = zeros(Float64,L)
        for loc in local_mags
            mags .+= loc
        end
        return mags
    end

    function connected_correlations_sector(ψ::AbstractVector{T}, p::SpinParams, 
                                            states::Vector{UInt64}) where {T<:Number}
        L, N = p.L, length(ψ)
        nth = nthreads()
        local_sz = [zeros(Float64, N, L) for _ in 1:nth]

        Threads.@threads for idx in 1:N
            prob = abs2(ψ[idx])
            if prob != 0.0
                state = states[idx]
                tid = threadid()
                loc = local_sz[tid]
                for i in 0:(L-1)
                    loc[idx,i+1] = sz_value(bit_at(state,i)) * prob
                end
            end
        end

        szvals = zeros(Float64, N, L)
        for loc in local_sz
            szvals .+= loc
        end

        S_i = sum(szvals, dims=1)[:]
        C_r = zeros(Float64,L)
        for r in 0:(L-1)
            tmp = 0.0
            for i in 1:L
                j = mod1(i+r,L)
                tmp += sum(szvals[:,i] .* szvals[:,j]) - S_i[i]*S_i[j]
            end
            C_r[r+1] = tmp / L
        end
        return C_r
    end

    function structure_factor_Sq_sector(ψ::AbstractVector{T}, p::SpinParams, 
                                        states::Vector{UInt64}) where {T<:Number}

        C_r = connected_correlations_sector(ψ, p, states)
        S_q = fft(C_r)
        qlist = [2π*(n-1)/p.L for n in 1:p.L]
        Sq_dict = Dict{Float64,Float64}()
        for n in 1:p.L
            Sq_dict[qlist[n]] = real(S_q[n])
        end
        return Sq_dict
    end



    # -------------------------------------------------------------------
    # KPM for T=0 Dynamical Correlation Functions
    # -------------------------------------------------------------------

    """
    kpm_dynamical_correlation(ψ::AbstractVector{T}, operator_i, operator_j, ω_range, p; 
                              n=300, ϵ=0.1, states, idxmap)
    Calculates the T=0 dynamical correlation function S(ω) = <ψ| A δ(ω - (H-E₀)) B |ψ> 
        using KPM.
    operator_i and operator_j are functions that apply operators A and B to a state vector.
    p is the SpinParams object defining the Hamiltonian.
    n is the number of Chebyshev moments to compute.
    ϵ is a small broadening parameter.
    states and idxmap are optional parameters for working in a specific symmetry sector.

    ψ in KPM defined as the ground state |GS⟩.

    Returns ω_range, S_ω.
    """
    function kpm_dynamical_correlation(ψ::AbstractVector{T}, operator_i, operator_j,
                                    ω_range::Vector{Float64}, p::SpinParams;
                                    n::Int=300, ϵ::Float64=0.1,
                                    states=nothing, idxmap=nothing) where T<:Number

        # 1. Estimate energy bounds for rescaling
        E_min, E_max = estimate_energy_bounds(p, states, idxmap)
        a = (E_max - E_min) / 2 * 0.99
        b = (E_max + E_min) / 2

        # 2. Apply operator B to the ground state: |ϕ₀⟩ = B |GS⟩
        ϕ₀ = operator_j(gs_vector, p, states, idxmap)
        
        # 3. Get the Jackson kernel coefficients
        jackson_kernel = get_jackson_kernel(n) # Precompute g_n coefficients

        # 4. Compute Chebyshev moments μ_n = ⟨ϕ₀| T_n(Ḣ) |ϕ₀⟩
        μ_n = compute_chebyshev_moments(ϕ₀, n, a, b, apply_H_sector!, p, states, idxmap)
        
        # 5. Apply the kernel to the moments
        μ_n .*= jackson_kernel

        # 6. Reconstruct the spectral function S(ω) on the requested ω_range
        S_ω = zeros(Float64, length(ω_range))
        
        for (i, ω) in enumerate(ω_range)
            # Rescale energy
            x = (ω - b) / a # Note: ω here is already (E - E₀), so we use the same rescaling
            # Evaluate the Chebyshev series
            S_ω[i] = evaluate_chebyshev_series(μ_n, x, a)
        end

        return ω_range, S_ω
    end

    """
    compute_chebyshev_moments(ψ0, n, a, b, applyH!, p, states, idxmap)
    Computes the moments μ_n = ⟨ψ0| T_n(Ḣ) |ψ0⟩ for the Chebyshev expansion.
    Uess the same recurrence as time evolution but only calculates the overlap.
    """
    function compute_chebyshev_moments(ψ0::Vector{T}, n::Int, a::Float64, b::Float64,
                                    applyH!, p::SpinParams, states, idxmap) where T<:Number

        moments = zeros(Float64, n)
        dim = length(ψ0)
        
        # Initialize states for recurrence
        ϕ_prev = copy(ψ0)
        # Normalize initial state? Moments depend on normalization.
        norm_ψ0 = norm(ψ0)
        ϕ_prev ./= norm_ψ0
        
        # Apply rescaled H to get ϕ_curr = T₁(Ḣ)ψ0
        ϕ_curr = apply_rescaled_H(ϕ_prev, applyH!, p, states, idxmap, a, b)
        
        # Calculate moments
        moments[1] = dot(ϕ_prev, ϕ_prev) * norm_ψ0^2  # μ₀
        moments[2] = dot(ϕ_prev, ϕ_curr) * norm_ψ0^2  # μ₁
        
        for k in 2:n-1
            # Chebyshev recurrence
            ϕ_next = 2 * apply_rescaled_H(ϕ_curr, applyH!, p, states, idxmap, a, b) - ϕ_prev
            
            # Calculate moment μ_k = ⟨ψ0| T_k(Ḣ) |ψ0⟩
            moments[k+1] = dot(ϕ_prev, ϕ_next) * norm_ψ0^2 # Using ϕ_prev which is T_{k-1}
            
            # Cycle the states
            ϕ_prev, ϕ_curr = ϕ_curr, ϕ_next
        end
        
        return moments
    end

    """
    get_jackson_kernel(n)
    Returns the Jackson kernel coefficients gₙ for a given expansion order n.
    """
    function get_jackson_kernel(n::Int)
        g = zeros(Float64, n)
        for k in 0:n-1
            Δ = π / (n+1)
            g[k+1] = ((n - k + 1) * cos(Δ * k) + sin(Δ * k) / tan(Δ)) / (n + 1)
        end
        return g
    end

    """
    evaluate_chebyshev_series(μ_n, x, a)
    Evaluates the KPM series sum at a given point x.
    """
    function evaluate_chebyshev_series(μ_n::Vector{Float64}, x::Float64, a::Float64)
        n = length(μ_n)
        total = μ_n[1] * 1.0 # n=0 term: T₀(x) = 1
        
        if n > 1
            total += μ_n[2] * x # n=1 term: T₁(x) = x
        end
        
        # Use recurrence for higher orders
        T_prev = 1.0
        T_curr = x
        for k in 2:n-1
            T_next = 2 * x * T_curr - T_prev # T_{k+1}(x)
            total += μ_n[k+1] * T_next
            T_prev, T_curr = T_curr, T_next
        end
        
        # Include the KPM density factor
        return total / (π * sqrt(1 - x^2)) * (2/a) # The (2/a) factor comes from the change of variables
    end

end # module


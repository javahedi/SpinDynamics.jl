module KPM

    """
    KPM gives (site-resolved)
    S_AB (ω) = ⟨ψ0|A^†_i δ(ω−(H-E0))B|ψ0⟩ ---> kpm_dynamical_correlation
    spin spectral function:
     S_ij (ω) = C_ij(ω) = ∫dt e^{iωt} ⟨S_i^α(0) S_j^α(t)⟩, --> kpm_correlation_matrix
     autocorrelation of the spin at site i 
     i.e. A=B=S_z
     C_ii(ω) = ∫dt e^{iωt} ⟨S_i^z(0) S_i^z(t)⟩ ---> kpm_dynamical_correlation with A=B=S_z

    To get the dynamical structure factor  S(q,ω)
    we  must do the Fourier transform over sites to get momentum dependent
    :: Shows collective modes and dispersion.
    S(q,ω) =1/N ∑_{ij} ​e^{-iq⋅(ri-rj)} C_ij (ω) ---> kpm_structure_factor_Sq


    """

    using LinearAlgebra
    using SpecialFunctions: besselj
    using ...SpinModel#: Model
    using ...Basis#: SpinParams, build_sector_basis
    using ...Hamiltonian#: apply_H_sector!, apply_H_full!
    using ...Observables#: magnetization_per_site_sector, structure_factor_Sq_sector,
                        # magnetization_per_site, structure_factor_Sq
    using ..Chebyshev #: apply_rescaled_H!, estimate_energy_bounds  # Reuse existing functionality
    using ...InitialStates#: domain_wall_state_sector, domain_wall_state_full
    
    export kpm_dynamical_correlation, kpm_dynamical_correlation_matrix,
            run_kpm_dynamical, kpm_correlation_matrix, Sqω


    # -------------------------------------------------------------------
    # KPM for T=0 Dynamical Correlation Functions
    # -------------------------------------------------------------------

  
    """
        get_rescaling_params(N::Int, apply_H!, model::SpinModel.Model; lanc_m=80)
    Estimate the rescaling parameters a,b to map H to Ḣ = (H - b)/a in [-1,1].
    Uses Lanczos to estimate the extremal eigenvalues of H.
    """
    function get_rescaling_params(apply_H!, model::SpinModel.Model; lanc_m::Int=80)
        E_min, E_max = estimate_energy_bounds(apply_H!, model; lanc_m=lanc_m)
        a = (E_max - E_min) / 2 * 0.9
        b = (E_max + E_min) / 2
        return a, b
    end



    """
        kpm_dynamical_correlation(ψ::AbstractVector{T}, operator_A, operator_B, 
                                ω_range, model::SpinModel.Model; 
                                n=300, ϵ=0.1)

    Calculates the T=0 dynamical correlation function S(ω) = ⟨ψ| A† δ(ω - (H-E₀)) B |ψ⟩ 
    using KPM.

    Arguments:
    - ψ: ground state vector
    - operator_A, operator_B: functions that apply operators A and B to a state vector
    - ω_range: array of frequencies to evaluate
    - model: SpinModel.Model instance defining the Hamiltonian
    - n: number of Chebyshev moments to compute
    - ϵ: small broadening parameter

    Returns:
    - S_ω
    """
    function kpm_dynamical_correlation(ψ::AbstractVector{T}, operator_A, operator_B,
                                    ω_range::AbstractVector{Float64}, 
                                    applyH!, model::SpinModel.Model;
                                    n::Int=300, ϵ::Float64=0.1,
                                    a::Union{Nothing,Float64}=nothing,
                                    b::Union{Nothing,Float64}=nothing) where T<:Number

        # 1. If a,b not provided, compute them
        if a === nothing || b === nothing
            a, b = get_rescaling_params(applyH!, model; lanc_m=n)
        end

        # 2. Apply operator B to the ground state: |ϕ⟩ = B |ψ⟩
        ϕ = operator_B(ψ, model)
        
        # 3. Apply operator A to the ground state: |χ⟩ = A |ψ⟩
        χ = operator_A(ψ, model)
        
        # 4. Get the Jackson kernel coefficients
        jackson_kernel = get_jackson_kernel(n)
        
        # 5. Compute Chebyshev moments μ_n = ⟨χ| T_n(Ḣ) |ϕ⟩
        μ_n = compute_cross_chebyshev_moments(χ, ϕ, n, a, b, applyH!, model)
        
        # 6. Apply the kernel to the moments
        μ_n .*= jackson_kernel

        # 7. Reconstruct the spectral function S(ω) on the requested ω_range
        S_ω = zeros(Float64, length(ω_range))
        
        for (i, ω) in enumerate(ω_range)
            # Rescale energy
            x = (ω - b) / a
            
            # Evaluate the Chebyshev series
            S_ω[i] = evaluate_chebyshev_series(μ_n, x, a)
        end


        return S_ω = max.(S_ω, 0.0)  # Ensure non-negativity
    end

    """
        compute_cross_chebyshev_moments(χ, ϕ, n, a, b, applyH!, model::SpinModel.Model)

    Computes the cross moments μ_n = ⟨χ| T_n(Ḣ) |ϕ⟩ for the Chebyshev expansion.
    """
    function compute_cross_chebyshev_moments(χ::AbstractVector{T}, ϕ::AbstractVector{T}, 
                                        n::Int, a::Float64, b::Float64, applyH!,
                                         model::SpinModel.Model) where T<:Number

        moments = zeros(Float64, n)
        N = length(ϕ)
        
        # Initialize states for recurrence
        ϕ_prev = copy(ϕ)
        norm_ϕ = norm(ϕ)
        ϕ_prev ./= norm_ϕ
        
        # Workspace vectors
        ϕ_curr = similar(ϕ)
        ϕ_next = similar(ϕ)
        temp = similar(ϕ)
        
        # Apply rescaled H to get ϕ_curr = T₁(Ḣ)ϕ
        apply_rescaled_H!(ϕ_curr, ϕ_prev, applyH!, model, a, b)
        
        # Calculate moments
        #moments[1] = dot(χ, ϕ_prev) * norm_ϕ  # μ₀ = ⟨χ|T₀(Ḣ)|ϕ⟩
        #moments[2] = dot(χ, ϕ_curr) * norm_ϕ  # μ₁ = ⟨χ|T₁(Ḣ)|ϕ⟩

        moments[1] = dot(conj(χ), ϕ_prev) * norm_ϕ  # μ₀ = ⟨χ|T₀(Ḣ)|ϕ⟩
        moments[2] = dot(conj(χ), ϕ_curr) * norm_ϕ  # μ₁ = ⟨χ|T₁(Ḣ)|ϕ⟩


        
        for k in 2:n-1
            # Chebyshev recurrence: ϕ_next = 2 * Ḣ * ϕ_curr - ϕ_prev
            apply_rescaled_H!(temp, ϕ_curr, applyH!, model, a, b)
            @. ϕ_next = 2 * temp - ϕ_prev
            
            # Calculate moment μ_k = ⟨χ| T_k(Ḣ) |ϕ⟩
            #moments[k+1] = real(dot(χ, ϕ_next)) * norm_ϕ
            moments[k+1] = real(dot(conj(χ), ϕ_next)) * norm_ϕ

            
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
    function evaluate_chebyshev_series(μ_n::AbstractVector{Float64}, x::Float64, a::Float64)
        n = length(μ_n)
        
        # Handle edge cases
        if abs(x) >= 1.0
            return 0.0  # Outside the Chebyshev domain
        end
        
        total = μ_n[1] * 1.0  # n=0 term: T₀(x) = 1
        
        if n > 1
            total += μ_n[2] * x  # n=1 term: T₁(x) = x
        end
        
        # Use recurrence for higher orders
        T_prev = 1.0
        T_curr = x
        for k in 2:n-1
            T_next = 2 * x * T_curr - T_prev  # T_{k+1}(x)
            total += μ_n[k+1] * T_next
            T_prev, T_curr = T_curr, T_next
        end
        
        # Include the KPM density factor
        return total / (π * sqrt(1 - x^2)) * (2/a)
    end




    function kpm_correlation_matrix(ψ::AbstractVector, ω_range::AbstractVector{Float64},
                                    applyH!, model::SpinModel.Model; n::Int=300, ϵ::Float64=0.1,
                                    opA_type_a::Symbol=:z, opB_type_b::Symbol=:z)

        N = model.L
        C = zeros(Float64, N, N, length(ω_range))

        # compute rescaling params ONCE
        a, b = get_rescaling_params(applyH!, model)

        for i in 1:N, j in 1:N
            opA = create_spin_operator(i, opA_type_a)
            opB = create_spin_operator(j, opB_type_b)
            S_ω = kpm_dynamical_correlation(ψ, opA, opB, ω_range, applyH!, model;
                                            n=n, ϵ=ϵ, a=a, b=b)
                                           
            C[i, j, :] .= abs.(S_ω)
            

        end
        return C
    end



    function Sqω(C::Array{Float64,3}, q::Float64, positions::Vector{Float64})
        N = length(positions)
        ω_len = size(C, 3)
        S = zeros(Float64, ω_len)
        for i in 1:N, j in 1:N
            phase = exp(-im * q * (positions[i] - positions[j]))
            S .+= (1/N) .* real(phase .* C[i,j,:])
        end
        return S
    end




    # High-level wrapper functions
    function run_kpm_dynamical(model::SpinModel.Model, ω_range;
                                    opA_type_a::Symbol=:z, 
                                    opB_type_b::Symbol=:z,
                                    n::Int=300, ϵ::Float64=0.1)

    
        
        ψ0 = domain_wall_state(model)
        ψ0 = ComplexF64.(ψ0)
        ψ0 ./= norm(ψ0)

        return kpm_correlation_matrix(ψ0, ω_range, apply_H_sector!, model; n=n, ϵ=ϵ,
                                    opA_type_a=opA_type_a, opB_type_b=opB_type_b)
    end


end # module KPM
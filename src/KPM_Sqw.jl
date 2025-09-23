module KPM_Sqw

    using LinearAlgebra
    using Base.Threads
    using ..SpinModel
    using ..Hamiltonian
    using ..Lanczos


    export kpm_sqw


    """
        get_rescaling_params(N::Int, apply_H!, model::SpinModel.Model; lanc_m=80)
    Estimate the rescaling parameters a,b to map H to Ḣ = (H - b)/a in [-1,1].
    Uses Lanczos to estimate the extremal eigenvalues of H.
    """
    function get_rescaling_params(apply_H!, model::SpinModel.Model; lanc_m::Int=80)
        E_min, E_max = estimate_energy_bounds(apply_H!, model; lanc_m=lanc_m)
        a = (E_max - E_min) / 2 * 0.99
        b = (E_max + E_min) / 2
        return a, b
    end
   




    function kpm_sw(phi::AbstractVector{ComplexF64},
                            apply_H!,
                            model::SpinModel.Model,
                            ω_range::AbstractVector{Float64};
                            a::Float64, b::Float64,
                            kpm_m::Int=200, 
                            kernel::Symbol=:jackson)

        # Compute moments correctly

        μ = compute_chebyshev_moments(apply_H!, phi, kpm_m, a, b, model)
        
        # Apply kernel
        g = get_kernel(kpm_m, kernel)
        
        μ .*= g  # μ_n *= g_n
        
        # Reconstruct spectrum
        S = zeros(Float64, length(ω_range))
        
        for (iω, ω) in enumerate(ω_range)
            x = (ω - b) / a
            x = clamp(x, -0.999, 0.999)  # Avoid boundaries
            
            # Chebyshev polynomials at this x
            T = zeros(Float64, kpm_m)
            T[1] = 1.0
            if kpm_m >= 2
                T[2] = x
            end
            for n in 3:kpm_m
                T[n] = 2.0 * x * T[n-1] - T[n-2]
            end
            
            # KPM reconstruction formula
            sum_val = μ[1] * T[1]
            for n in 2:kpm_m
                sum_val += 2.0 * μ[n] * T[n]
            end
            
            denom = π * sqrt(1.0 - x^2)
            S[iω] = max(0.0, sum_val / denom)  # Ensure non-negative
        end
        
        return S
    end

    function compute_chebyshev_moments(apply_H!, phi, M, a, b, model)
        mu = zeros(Float64, M)   # force real moments

        v_prev = copy(phi)              # T₀|phi⟩
        v_curr = similar(phi)           # T₁|phi⟩
        v_next = similar(phi)

        # μ₀ = ⟨phi|phi⟩
        mu[1] = real(dot(conj(phi), v_prev))

        # v_curr = H'|phi⟩
        apply_rescaled_H!(v_curr, v_prev, apply_H!, model, a, b)
        mu[2] = real(dot(conj(phi), v_curr))

        for m in 2:M-1
            # v_next = 2 H' v_curr - v_prev
            apply_rescaled_H!(v_next, v_curr, apply_H!, model, a, b)
            @. v_next = 2.0 * v_next - v_prev

            mu[m+1] = real(dot(conj(phi), v_next))

            # (optional safety renormalization)
            nv = norm(v_next)
            if nv > 1e3
                #@warn "Renormalizing v_next at step $m, norm=$nv"
                v_next ./= nv
            end

            # rotate vectors
            v_prev, v_curr, v_next = v_curr, v_next, v_prev
        end

        return mu
    end


    function get_kernel(M, kernel)
        g = ones(Float64, M)
        if kernel == :jackson
            for n in 0:M-1
                g[n+1] = ((M - n + 1) * cos(π * n / (M+1)) + sin(π * n / (M+1)) * cot(π / (M+1))) / (M + 1)

            end
        elseif kernel == :lorentz
            λ = 3.0  # Lorentz damping parameter
            for n in 0:M-1
                g[n+1] = sinh(λ * (1 - n/M)) / sinh(λ)
            end
        end
        return g
    end

  
    """
    Debug function to check moment calculation
    """
    
    function debug_moments(phi::AbstractVector{ComplexF64},
                        apply_H!,
                        model::SpinModel.Model, ω_range::AbstractVector{Float64};
                        kpm_m::Int=10,
                        a::Union{Nothing,Float64}=nothing,
                        b::Union{Nothing,Float64}=nothing)
        println("=== CRITICAL KPM DEBUG ===")
        println("Rescaling: a = $a, b = $b")

        println("norm(phi) = ", norm(phi))
        
        # Check if ω_range maps to reasonable x values
        x_min = (minimum(ω_range) - b) / a
        x_max = (maximum(ω_range) - b) / a
        println("x range in [-1,1]: $x_min to $x_max")
        
        if abs(x_min) > 1.1 || abs(x_max) > 1.1
            println("❌ WARNING: x outside [-1,1]! KPM will fail.")
        end
        
        # Check moment growth
        μ = compute_chebyshev_moments( apply_H!, phi, kpm_m, a, b, model)
        println("First 10 moments: ", μ[1:min(10, length(μ))])
        
        # Check if moments are exploding
        if maximum(abs.(μ)) > 1e6
            println("❌ MOMENTS EXPLODING: Max moment = $(maximum(abs.(μ)))")
        end
        
        # Test reconstruction at a few points
        test_ω = [ω_range[1], ω_range[end÷2], ω_range[end]]
        for ω in test_ω
            x = (ω - b) / a
            denom = π * sqrt(1.0 - x^2)
            println("ω=$ω, x=$x, 1/denom=$(1/denom)")
        end
    end


    function kpm_sqw(ψ0::AbstractVector{<:Number}, 
                            model::SpinModel.Model,
                            q_list::AbstractVector{Float64}, 
                            ω_range::AbstractVector{Float64};
                            a::Union{Nothing,Float64}=nothing,
                            b::Union{Nothing,Float64}=nothing,
                            kpm_m::Int=200,
                            kernel::Symbol=:jackson)
        
        ψ0c = ComplexF64.(ψ0)
        Qn = length(q_list)
        W = length(ω_range)
        Smat = zeros(Float64, Qn, W)
        
        # Get rescaling parameters once
        if a === nothing || b === nothing
            a, b = get_rescaling_params(apply_H!, model)
        end
    
        
        # Better: Threaded like Lanczos version
        @threads for iq in 1:length(q_list)
            q = q_list[iq]
       
            
         
            phi = Sz_q_vector(model, ψ0c, q)
            phi ./= norm(phi)
            
            if norm(phi) == 0
                Smat[iq, :] .= 0.0
                continue
            end


            #if iq>1
            #    debug_moments(phi, apply_H!, model, ω_range;
            #                kpm_m=kpm_m,  a=a, b=b)
            #end
          
            
            S = kpm_sw(phi, apply_H!, model, ω_range; kpm_m=kpm_m, 
                            a=a, b=b, kernel=kernel)
            Smat[iq, :] .= S
        end
        
        return Smat
    end


end # module



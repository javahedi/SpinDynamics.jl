module LanczosSqw

    using LinearAlgebra
    using FFTW
    using Base.Threads

    using ..Hamiltonian
    using ..SpinModel
    using ..Lanczos

    export lanczos_sqw

   

    # ---------------------------------
    # Construct spectral density S(ω)
    # ---------------------------------
    function spectral_from_tridiagonal(α::Vector{Float64}, β::Vector{Float64},
                                    norm_phi::Float64, E0::Float64, 
                                    ω_range::AbstractVector{Float64};
                                    eta::Float64=0.05, broaden::Symbol=:lorentz)

        T = SymTridiagonal(α, β)
        eig = eigen(T)
        θ = eig.values
        Q = eig.vectors
        w = abs2.(Q[1, :]) .* (norm_phi^2)

        if broaden == :lorentz
            shifted = ω_range .- (θ .- E0)'  # W × m
            Lmat = (1/pi) .* (eta ./ (shifted.^2 .+ eta^2))
            S = vec(Lmat * w)
        elseif broaden == :gauss
            pref = 1/(sqrt(2*pi)*eta)
            shifted = ω_range .- (θ .- E0)'  # W × m
            Gmat = pref .* exp.(-(shifted.^2)./(2*eta^2))
            S = vec(Gmat * w)
        else
            error("unknown broadening: $broaden")
        end

        return S
    end

    
    # ---------------------------------
    # High-level wrapper: S(q, ω)
    # ---------------------------------
    function lanczos_sqw(ψ0::AbstractVector{<:Number},
                        model::SpinModel.Model,
                        q_list::AbstractVector{Float64},
                        ω_range::AbstractVector{Float64};
                        lanc_m::Int=200, eta::Float64=0.05, 
                        broaden::Symbol=:lorentz)

        ψ0c = ComplexF64.(ψ0)
        tmp = zeros(ComplexF64, length(ψ0c))
        apply_H!(tmp, ψ0c, model)
        E0 = real(dot(conj(ψ0c), tmp))

        Qn = length(q_list)
        W = length(ω_range)
        Smat = zeros(Float64, Qn, W)

        @threads for iq in 1:Qn
            q = q_list[iq]
            phi = Sz_q_vector(model, ψ0c, q)
            if norm(phi) == 0
                Smat[iq, :] .= 0.0
                continue
            end

            α, β, norm_phi = lanczos_tridiag(apply_H!, model, phi; lanc_m=lanc_m)
            S = spectral_from_tridiagonal(α, β, norm_phi, E0, ω_range; 
                                        eta=eta, broaden=broaden)
            Smat[iq, :] .= S
        end

        return Smat
    end

end # module

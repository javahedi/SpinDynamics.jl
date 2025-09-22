module Krylov


    """
    Krylov time-evolution submodule for SpinDynamics.
    Provides Krylov/Lanczos time evolution for both:
    - sector-restricted real vectors (Float64)
    - full Hilbert complex vectors (ComplexF64)
    """

    using LinearAlgebra
    # import sibling modules from parent (SpinDynamics)
    using ...SpinModel
    using ...Basis
    using ...Hamiltonian
    using ...InitialStates
    using ...Observables

    export krylov_time_evolve!, krylov_time_evolve
    export run_krylov, KrylovWorkspace

    # ---------------------------------------------------------
    # Krylov time evolution 
    # ---------------------------------------------------------
    struct KrylovWorkspace
        V::Vector{Vector{ComplexF64}}   # Krylov basis vectors
        α::Vector{ComplexF64}
        β::Vector{ComplexF64}
        w::Vector{ComplexF64}
        e1::Vector{ComplexF64}
    end

    function KrylovWorkspace(n::Int, m::Int)
        V = [zeros(ComplexF64, n) for _ in 1:m]
        α = zeros(ComplexF64, m)
        β = zeros(ComplexF64, m-1)
        w = zeros(ComplexF64, n)
        e1 = zeros(ComplexF64, m)
        return KrylovWorkspace(V, α, β, w, e1)
    end


    """
        krylov_time_evolve!(ψ_out, ψ_in, dt, applyH!, model, ws; m=30)
    Advance the state vector `ψ_in` by time `dt` using Krylov–Lanczos time evolution,
    writing the result into `ψ_out`.

    - `ψ_out`, `ψ_in` are `Vector{ComplexF64}` of length N
    - `applyH!(out, vec, model)` applies H·vec into out
    - `model`: SpinModel.Model
    - `ws`: KrylovWorkspace (preallocated)
    - `m`: Krylov subspace dimension

    """
    function krylov_time_evolve!(ψ_out::Vector{ComplexF64},
                                ψ_in::Vector{ComplexF64},
                                dt::Float64,
                                applyH!, model::SpinModel.Model,
                                ws::KrylovWorkspace;
                                kry_m::Int=30)

        n = length(ψ_in)
        @assert length(ψ_out) == n
        @assert length(ws.V) ≥ kry_m

        V, α, β, w, e1 = ws.V, ws.α, ws.β, ws.w, ws.e1

        norm0 = norm(ψ_in)
        if norm0 == 0
            copyto!(ψ_out, ψ_in)
            return ψ_out
        end

        # first Krylov vector
        copyto!(V[1], ψ_in)
        V[1] ./= norm0

        m_eff = kry_m
        for j in 1:kry_m
           
            applyH!(w, V[j], model)
            

            α[j] = dot(V[j], w)
            w .-= α[j] .* V[j]
            if j > 1
                w .-= β[j-1] .* V[j-1]
            end

            if j < kry_m
                β[j] = norm(w)
                if abs(β[j]) < 1e-14
                    m_eff = j
                    break
                end
                copyto!(V[j+1], w)
                V[j+1] ./= β[j]
            end
        end

        # reduced matrix
        TR = SymTridiagonal(view(α, 1:m_eff), view(β, 1:m_eff-1))
        eig = eigen(Hermitian(TR))
        D, Q = eig.values, eig.vectors

        fill!(e1, 0)           # zero out instead of mul!
        e1[1] = norm0
        y = Q * (Diagonal(exp.(-1im .* D .* dt)) * (Q' * e1))

        # reconstruct ψ_out
        fill!(ψ_out, 0)
        for k in 1:m_eff
            @. ψ_out += y[k] * V[k]
        end

        ψ_out ./= norm(ψ_out)
        return nothing
    end


    
    
    """
        krylov_time_evolve(ψ0, t, applyH!, model::SpinModel.Model; m=30)

        - ψ0: Vector{ComplexF64} (sector basis)
        - applyH!: function of signature (out, vec, p, states, idxmap)
        - model: SpinModel.Model

        NOTE: 
        If dt is fixed → you can precompute Q, D once and reuse.
        If only dt changes → you can reuse Q and just update exp.(-1im*D*dt). 
        That avoids the eigen call each step.
        
        """
    function krylov_time_evolve(ψ0::AbstractVector{T}, dt::Float64,
                applyH!, model::SpinModel.Model; kry_m::Int=30) where T<:Number

            n = length(ψ0)
            V = Vector{Vector{T}}(undef, kry_m)
            α = zeros(ComplexF64, kry_m)  # CHANGED: Use ComplexF64
            β = zeros(ComplexF64, kry_m-1)  # CHANGED: Use ComplexF64
            w = zeros(T, n)

            norm0 = norm(ψ0)
            if norm0 == 0
                return copy(ψ0)
            end
            V[1] = copy(ψ0) / norm0

            m_eff = kry_m
            for j in 1:kry_m
                applyH!(w, V[j], model)

                α[j] = dot(V[j], w)  # CHANGED: Use full complex dot product
                w .-= α[j] .* V[j]
                if j > 1
                    w .-= β[j-1] .* V[j-1]
                end
                if j < kry_m
                    β[j] = norm(w)
                    if abs(β[j]) < 1e-14
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
            # CHANGED: Use Tridiagonal for complex values
            TR = Tridiagonal(β[1:(m_eff-1)], α[1:m_eff], β[1:(m_eff-1)])
            eig = eigen(Matrix(TR))  # CHANGED: Convert to matrix for complex eigen

            D = eig.values
            Q = eig.vectors
            U_T = Q * Diagonal(exp.(-1im .* D .* dt)) * Q'
            e1 = zeros(ComplexF64, m_eff); e1[1] = norm0  # CHANGED: ComplexF64
            y = U_T * e1

            # reconstruct complex ψt
            ψt = zeros(ComplexF64, n)  # CHANGED: ComplexF64
            for k in 1:m_eff
                ψt .+= y[k] .* V[k]
            end

            ψt ./= norm(ψt)
            return ψt
    end
    


    # ---------------------------------------------------------
    # High-level wrapper: Krylov (sector)
    # ---------------------------------------------------------
    """
    run_krylov(model::SpinModel.Model; m=30)

    Builds sector basis, domain-wall initial state, runs Krylov evolution and returns observables.
    """
    function run_krylov(model::SpinModel.Model, dt::Float64; kry_m::Int=30)

      
        ψ0 = domain_wall_state(model)

        # Time evolution
        ψt = krylov_time_evolve(ComplexF64.(ψ0), dt, apply_H!, model, kry_m=kry_m)


        # Observables
        mags = magnetization_per_site_sector(ψt, model)
        Sq   = structure_factor_Sq_sector(ψt, model)
        return mags, Sq
    end

    
end # module

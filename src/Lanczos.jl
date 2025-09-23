module Lanczos


    using  LinearAlgebra
    using ..SpinModel
    using ..Hamiltonian


    export lanczos_extremal, lanczos_groundstate, 
            lanczos_tridiag, estimate_energy_bounds
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
    function lanczos_extremal(applyH!, model::SpinModel.Model; 
                            lanc_m::Int=100, tol::Float64=1e-12) 

        N =  length(model.states)  # Hilber dim
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


    function lanczos_groundstate(applyH!, model::SpinModel.Model;
                             lanc_m::Int=100, tol::Float64=1e-12,
                             orthogonalize_tol::Float64=1e-10)
    
        N = length(model.states)
        ψ0 = randn(Float64, N)
        ψ0 ./= norm(ψ0)

        α = zeros(Float64, lanc_m)
        β = zeros(Float64, lanc_m-1)
        V = Matrix{Float64}(undef, N, lanc_m)
        V[:,1] = ψ0

        w = similar(ψ0)
        
        m_actual = lanc_m

        for j in 1:lanc_m
            vj = view(V, :, j)
            applyH!(w, vj, model)

            # Full reorthogonalization for better stability
            if j > 1
                for k in 1:j-1
                    vk = view(V, :, k)
                    coeff = dot(vk, w)
                    w .-= coeff .* vk
                end
            end

            α[j] = real(dot(vj, w))

            if j == 1
                w .= w .- α[j] .* vj
            else
                w .= w .- α[j] .* vj .- β[j-1] .* view(V, :, j-1)
            end

            if j < lanc_m
                β[j] = norm(w)
                
                # Check for breakdown and numerical stability
                if β[j] < tol
                    m_actual = j
                    break
                end
                
                # Additional orthogonalization check
                for k in 1:j
                    vk = view(V, :, k)
                    overlap = abs(dot(vk, w/β[j]))
                    if overlap > orthogonalize_tol
                        w .-= dot(vk, w) .* vk
                        β[j] = norm(w)
                        if β[j] < tol
                            m_actual = j
                            break
                        end
                    end
                end
                
                V[:,j+1] = w / β[j]
            end
        end

        # Use actual dimension reached
        α_actual = α[1:m_actual]
        β_actual = β[1:min(m_actual-1, length(β))]
        V_actual = V[:, 1:m_actual]

        TR = SymTridiagonal(α_actual, β_actual)
        evals, evecs = eigen(Hermitian(TR))
        
        Emin, idx = findmin(evals)
        y = evecs[:, idx]
        
        ψ_gs = V_actual * y
        ψ_gs ./= norm(ψ_gs)
        
        # Energy residual check for precision
        Hψ = similar(ψ_gs)
        applyH!(Hψ, ψ_gs, model)
        residual = norm(Hψ - Emin * ψ_gs)
        
        #@info "Lanczos precision" m_actual residual energy=Emin
        
        return Emin, ψ_gs
    end


    

     # Lanczos tridiagonalization (real symmetric) starting from v = phi.
    # Returns α (length m), β (length m-1), norm_phi.
    """
        lanczos_tridiag(applyH!, model, v; m=200, tol=1e-14)

    Lanczos tridiagonalization of H with starting vector v (not normalized).
    - applyH!(out, ψ, model) must write H*ψ into out
    - v: starting vector (ComplexF64)
    Returns (α, β, norm_v)
    """
    function lanczos_tridiag(applyH!, model::SpinModel.Model, 
                            v::AbstractVector{ComplexF64};
                            lanc_m::Int=100, tol::Float64=1e-12)

        n = length(v)
        V = Vector{Vector{ComplexF64}}(undef, lanc_m)
        α = zeros(Float64, lanc_m)
        β = zeros(Float64, lanc_m-1)

        # workspace
        w = zeros(ComplexF64, n)

        normv = norm(v)
        if normv == 0
            error("starting vector has zero norm")
        end

        V[1] = copy(v) / normv
        m_eff = lanc_m

        for j in 1:lanc_m-1
            applyH!(w, V[j], model)               # w = H * V[j]
            α[j] = real(dot(conj(V[j]), w))       # α_j

            # w = w - α_j V[j] - β_{j-1} V[j-1]
            @. w -= α[j] * V[j]
            if j > 1
                @. w -= β[j-1] * V[j-1]
            end

            β[j] = norm(w)
            if β[j] < tol
                m_eff = j
                break
            end

            V[j+1] = copy(w / β[j])
        end

        # last α (if didn't fill last slot)
        if m_eff == lanc_m
            applyH!(w, V[lanc_m], model)
            α[lanc_m] = real(dot(conj(V[lanc_m]), w))
        else
            α = α[1:m_eff]
            β = β[1:m_eff-1]
        end

        return α, β, normv
    end

    
    """
        estimate_energy_bounds(applyH!, N::Int, model::SpinModel.Model; m=80)

    Estimate the minimum and maximum eigenvalues (Emin, Emax) of the Hamiltonian 
    using Lanczos applied to `ψ0`.
    """
    function estimate_energy_bounds(applyH!, model::SpinModel.Model;
                                    lanc_m::Int=80)
        # Estimate Emax
        _, Emax    = lanczos_extremal(applyH!, model; lanc_m=lanc_m)
        
        # Define negative Hamiltonian
        function apply_H_neg!(out, ψ, model)
                applyH!(out, ψ, model)
                @. out = -out
                return out
        end

        _, Emax_neg = lanczos_extremal(apply_H_neg!, model; lanc_m=lanc_m)
        Emin = -Emax_neg
        
        return Emin, Emax
    end

    
end # modele
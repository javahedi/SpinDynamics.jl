module Observables

    using FFTW
    using Base.Threads
    using ..Hamiltonian
    using ..SpinModel

    export magnetization_per_site, structure_factor_Sq, connected_correlations


    # --------------------------------------------------
    # Magnetization per site 
    # --------------------------------------------------
    function magnetization_per_site(ψ::AbstractVector{T}, model::SpinModel.Model) where T<:Number
        L, N = model.L, length(ψ)
        nth = nthreads()
        local_mags = [zeros(Float64, L) for _ in 1:nth]

        Threads.@threads for idx in 1:N
            prob = abs2(ψ[idx])
            if prob != 0.0
                state = model.mode == :full ? UInt64(idx-1) : model.states[idx]
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
    function connected_correlations(ψ::AbstractVector{T}, model::SpinModel.Model) where T<:Number
        L = model.L
        N = length(ψ)
        nth = nthreads()

        # Thread-local accumulators for <S_i S_j> contributions
        local_sz = [zeros(Float64, L, L) for _ in 1:nth]
        local_Si = [zeros(Float64, L) for _ in 1:nth]

        Threads.@threads for idx in 1:N
            amp2 = abs2(ψ[idx])
            if amp2 == 0.0
                continue
            end
            tid = threadid()
            szmat = local_sz[tid]
            Si = local_Si[tid]

            # Determine the state depending on mode
            state = model.mode == :full ? UInt64(idx-1) : model.states[idx]

            # Accumulate <S_i> and <S_i S_j> contributions
            for i in 0:(L-1)
                szi = sz_value(bit_at(state, i))
                Si[i+1] += amp2 * szi
                for j in 0:(L-1)
                    szmat[i+1,j+1] += amp2 * szi * sz_value(bit_at(state,j))
                end
            end
        end

        # Reduce across threads
        SzSz = zeros(Float64, L, L)
        S_i = zeros(Float64, L)
        for t in 1:nth
            SzSz .+= local_sz[t]
            S_i .+= local_Si[t]
        end

        # Connected correlations C_r = ⟨S_i S_{i+r}⟩ - ⟨S_i⟩⟨S_{i+r}⟩
        C_r = zeros(Float64, L)
        for r in 0:(L-1)
            tmp = 0.0
            for i in 1:L
                j = mod1(i+r, L)
                tmp += SzSz[i,j] - S_i[i]*S_i[j]
            end
            C_r[r+1] = tmp / L
        end

        return C_r
    end


    # --------------------------------------------------
    # Spin structure factor S(q)
    # --------------------------------------------------
    function structure_factor_Sq(ψ::AbstractVector{T}, model::SpinModel.Model) where {T<:Number}
        C_r = connected_correlations(ψ, model)
        S_q = fft(C_r)
        qlist = [2π*(n-1)/model.L for n in 1:model.L]
        Sq_dict = Dict{Float64,Float64}()
        for n in 1:model.L
            Sq_dict[qlist[n]] = real(S_q[n])
        end
        return Sq_dict
    end


end # module


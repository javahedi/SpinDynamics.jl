module Hamiltonian


    export apply_H!, apply_rescaled_H!
    
    export bit_at, sz_value, flip_bits, create_spin_operator, Sz_q_vector

    using ..SpinModel
    
    using Base.Threads


    # Bit helpers
    """
    bit_at(state, i) -> 0 or 1
    Return the value of the i-th spin in the bitstring `state`.
    """
    # Bit helpers with improved performance
    @inline function bit_at(state::UInt64, i::Int)
        return (state >> i) & 0x1
    end

    @inline function sz_value(bit::UInt64)
        return bit == 1 ? 0.5 : -0.5
    end

    @inline function flip_bits(state::UInt64, i::Int, j::Int)
        return state ⊻ (UInt64(1) << i) ⊻ (UInt64(1) << j)
    end


    """
        create_spin_operator(site::Int, op_type::Symbol)

    Create a spin operator function for a specific site.
    op_type can be :z, :plus, :minus, :x, :y

    usage:
        op = create_spin_operator(1, :z)
        new_state = op(ψ, model)

        # Create spin operators on site 2
        Sz2 = create_spin_operator(2, :z)
        # Apply them to ψ0
        ψ_sz2 = Sz2(ψ0, model)
        # Compute expectation values
        ⟨Sz2⟩ = real(dot(ψ0, ψ_sz2))   # ⟨ψ| S_z(2) |ψ⟩
    """
    function create_spin_operator(site::Int, op_type::Symbol)
        site >= 1 || throw(ArgumentError("site must be at least 1"))

        op_type in (:z, :plus, :minus, :x, :y) ||
            throw(ArgumentError(
                "unsupported spin operator: $op_type; expected :z, :plus, :minus, :x, or :y"
            ))

        bit_pos = site - 1

        function operator(ψ::AbstractVector{T}, model::SpinModel.Model) where T
            site <= model.L ||
                throw(ArgumentError("site $site is outside the model with L=$(model.L)"))

            length(ψ) == length(model.states) ||
                throw(DimensionMismatch(
                    "state vector has length $(length(ψ)), expected $(length(model.states))"
                ))

            if model.mode === :sector && op_type !== :z
                throw(ArgumentError(
                    "operator $op_type changes total magnetization and cannot be applied " *
                    "within a fixed-nup sector"
                ))
            end

            result = zeros(T, length(ψ))

            for (idx, state) in enumerate(model.states)
                current_bit = bit_at(state, bit_pos)
                
                if op_type == :z
                    # S_z operator: diagonal
                    result[idx] = sz_value(current_bit) * ψ[idx]
                    
                elseif op_type == :plus
                    # S⁺ operator: flip 0→1
                    if current_bit == 0
                        new_state = state ⊻ (UInt64(1) << bit_pos)
                        if haskey(model.idxmap, new_state)
                            result[model.idxmap[new_state]] += ψ[idx]
                        end
                    end
                    
                elseif op_type == :minus
                    # S⁻ operator: flip 1→0
                    if current_bit == 1
                        new_state = state ⊻ (UInt64(1) << bit_pos)
                        if haskey(model.idxmap, new_state)
                            result[model.idxmap[new_state]] += ψ[idx]
                        end
                    end
                    
                elseif op_type == :x
                    # S_x = (S⁺ + S⁻)/2
                    if current_bit == 0
                        new_state = state ⊻ (UInt64(1) << bit_pos)
                        if haskey(model.idxmap, new_state)
                            result[model.idxmap[new_state]] += 0.5 * ψ[idx]
                        end
                    else
                        new_state = state ⊻ (UInt64(1) << bit_pos)
                        if haskey(model.idxmap, new_state)
                            result[model.idxmap[new_state]] += 0.5 * ψ[idx]
                        end
                    end
                    
                elseif op_type == :y
                    # S_y = (S⁺ - S⁻)/(2i)
                    if current_bit == 0
                        new_state = state ⊻ (UInt64(1) << bit_pos)
                        if haskey(model.idxmap, new_state)
                            result[model.idxmap[new_state]] += -0.5im * ψ[idx]
                        end
                    else
                        new_state = state ⊻ (UInt64(1) << bit_pos)
                        if haskey(model.idxmap, new_state)
                            result[model.idxmap[new_state]] += 0.5im * ψ[idx]
                        end
                    end
                end
            end
            
            return result
        end
        
        return operator
    end

    # ------------------------
    # Full Hilbert space H
    # function apply_H!(out::AbstractVector{T}, ψ::AbstractVector{T}, 
    #                         model::SpinModel.Model) where T <: Number
    #     L = model.L
    #     N = length(ψ)
    #     #@assert N == (1 << L) "State vector size doesn't match Hilbert space dimension"

    #     fill!(out, zero(T))

    #     # Precompute thread-local storage
    #     #nth = nthreads()
    #     #thread_buffers = [zeros(T, N) for _ in 1:nth]
    #     thread_buffers = [zeros(T, N) for _ in 1:Threads.maxthreadid()]
        

    #     Threads.@threads for idx in 1:N
    #         tid = threadid()
    #         local_out = thread_buffers[tid]
    #         amp = ψ[idx]

    #         if iszero(amp)
    #             continue
    #         end

    #         # choose state depending on mode
    #         state = model.mode == :full ? UInt64(idx-1) : model.states[idx]
    #         diag = zero(T)

    #         # diagonal field
    #         # Diagonal terms
    #         for i in 1:L
    #             diag += T(model.onsite_field[i] * sz_value(bit_at(state, i-1)))
    #         end

    #         # ZZ terms
    #         for (i, j, Jz) in model.zz_list
    #             diag += T(Jz * sz_value(bit_at(state, i-1)) * sz_value(bit_at(state, j-1)))
    #         end

    #         @inbounds local_out[idx] += diag * amp

    #         # hopping terms
    #         for (i, j, Jxy) in model.hopping_list
    #             bit_i = bit_at(state, i-1)
    #             bit_j = bit_at(state, j-1)
                
    #             if bit_i != bit_j
    #                 new_state = flip_bits(state,i-1,j-1)
    #                 if model.mode == :full
    #                     new_idx = Int(new_state)+1
    #                     @inbounds local_out[new_idx] += T(Jxy)*amp
    #                 elseif model.mode == :sector
    #                     if haskey(model.idxmap, new_state)
    #                         new_idx = model.idxmap[new_state]
    #                         @inbounds local_out[new_idx] += T(Jxy)*amp
    #                     end
    #                 end
    #             end
    #         end
    #     end

    #     # Combine thread results
    #     # for i in 1:nth
    #     #     @inbounds out .+= thread_buffers[i]
    #     # end
    #     for buffer in thread_buffers
    #         @inbounds out .+= buffer
    #     end
        
    #     return out
    # end

    function apply_H!(
        out::AbstractVector{T},
        ψ::AbstractVector{T},
        model::SpinModel.Model,
    ) where T <: Number

        L = model.L
        N = length(ψ)

        @assert length(out) == N

        Threads.@threads for idx in 1:N
            state = model.mode === :full ? UInt64(idx - 1) : model.states[idx]

            # Diagonal contribution
            diag = zero(T)

            for i in 1:L
                diag += T(
                    model.onsite_field[i] *
                    sz_value(bit_at(state, i - 1))
                )
            end

            for (i, j, Jz) in model.zz_list
                diag += T(
                    Jz *
                    sz_value(bit_at(state, i - 1)) *
                    sz_value(bit_at(state, j - 1))
                )
            end

            value = diag * ψ[idx]

            # Off-diagonal XY contribution.
            # Each thread writes only to out[idx], so no thread-local
            # output buffers are required.
            for (i, j, Jxy) in model.hopping_list
                bit_i = bit_at(state, i - 1)
                bit_j = bit_at(state, j - 1)

                if bit_i != bit_j
                    new_state = flip_bits(state, i - 1, j - 1)

                    if model.mode === :full
                        new_idx = Int(new_state) + 1
                        @inbounds value += T(Jxy) * ψ[new_idx]

                    else
                        new_idx = get(model.idxmap, new_state, 0)

                        if new_idx != 0
                            @inbounds value += T(Jxy) * ψ[new_idx]
                        end
                    end
                end
            end

            @inbounds out[idx] = value
        end

        return out
    end




    """
        apply_rescaled_H!(out, ψ, applyH!, model, idxmap, a, b)

    Compute out .= (Hψ - b*ψ) / a, where Hψ is written by applyH!(out, ψ, p[, states, idxmap]).
    - `out` and `ψ` must be preallocated and have the same length.
    - This function performs all operations in-place and avoids temporaries.
    Returns `out`.
    """
    function apply_rescaled_H!(out::AbstractVector{T}, ψ::AbstractVector{T},
                            applyH!, model::SpinModel.Model, 
                            a::Float64, b::Float64) where T<:Number
        @assert length(out) == length(ψ)
        
        # Fill out with H * ψ via the user-provided applyH!
        applyH!(out, ψ, model)
        
        
        # In-place rescaling: out = (out - b * ψ) / a
        @inbounds for i in eachindex(out)
            out[i] = (out[i] - b * ψ[i]) / a
        end
        
        return out
    end


      # ------------------------------
    # Build S_q^z |psi0> vector (phi)
    # ------------------------------
    function Sz_q_vector(
        model::SpinModel.Model,
        psi0::AbstractVector{T},
        q::Float64,
    ) where T <: Number

        L = model.L
        N = length(psi0)

        normfact = inv(sqrt(L))
        phases = exp.(im * q * (0:(L - 1)))

        phi = zeros(ComplexF64, N)

        Threads.@threads for idx in 1:N
            state = model.mode === :full ?
                UInt64(idx - 1) :
                model.states[idx]

            sq = 0.0 + 0.0im

            for r in 0:(L - 1)
                sq += phases[r + 1] * sz_value(bit_at(state, r))
            end

            @inbounds phi[idx] =
                normfact * sq * ComplexF64(psi0[idx])
        end

        return phi
    end



end # module

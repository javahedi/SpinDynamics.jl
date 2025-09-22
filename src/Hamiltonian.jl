module Hamiltonian


    export apply_H!
    export bit_at, sz_value, flip_bits, create_spin_operator

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
        bit_pos = site - 1  # Convert to 0-based

        function operator(ψ::AbstractVector{T}, model::SpinModel.Model) where T

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
    function apply_H!(out::AbstractVector{T}, ψ::AbstractVector{T}, 
                            model::SpinModel.Model) where T <: Number
        L = model.L
        N = length(ψ)
        #@assert N == (1 << L) "State vector size doesn't match Hilbert space dimension"

        fill!(out, zero(T))

        # Precompute thread-local storage
        nth = nthreads()
        thread_buffers = [zeros(T, N) for _ in 1:nth]
        

        Threads.@threads for idx in 1:N
            tid = threadid()
            local_out = thread_buffers[tid]
            amp = ψ[idx]

            if iszero(amp)
                continue
            end

            # choose state depending on mode
            state = model.mode == :full ? UInt64(idx-1) : model.states[idx]
            diag = zero(T)

            # diagonal field
            # Diagonal terms
            for i in 1:L
                diag += T(model.onsite_field[i] * sz_value(bit_at(state, i-1)))
            end

            # ZZ terms
            for (i, j, Jz) in model.zz_list
                diag += T(Jz * sz_value(bit_at(state, i-1)) * sz_value(bit_at(state, j-1)))
            end

            @inbounds local_out[idx] += diag * amp

            # hopping terms
            for (i, j, Jxy) in model.hopping_list
                bit_i = bit_at(state, i-1)
                bit_j = bit_at(state, j-1)
                
                if bit_i != bit_j
                    new_state = flip_bits(state,i-1,j-1)
                    if model.mode == :full
                        new_idx = Int(new_state)+1
                        @inbounds local_out[new_idx] += T(Jxy)*amp
                    elseif model.mode == :sector
                        if haskey(model.idxmap, new_state)
                            new_idx = model.idxmap[new_state]
                            @inbounds local_out[new_idx] += T(Jxy)*amp
                        end
                    end
                end
            end
        end

        # Combine thread results
        for i in 1:nth
            @inbounds out .+= thread_buffers[i]
        end
        
        return out
    end



end # module

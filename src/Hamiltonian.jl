module Hamiltonian

export SpinParams, apply_H_full!, apply_H_sector!
export bit_at, sz_value, flip_bits

using Base.Threads
 


struct SpinParams
    L::Int
    hopping_list::Vector{Tuple{Int,Int,Float64}}
    onsite_field::Vector{Float64}
    zz_list::Vector{Tuple{Int,Int,Float64}}
end

# Bit helpers
"""
bit_at(state, i) -> 0 or 1
Return the value of the i-th spin in the bitstring `state`.
"""
@inline bit_at(state::UInt64,i::Int)=UInt64((state>>i)&1)
@inline sz_value(bit::UInt64)=bit==1 ? 0.5 : -0.5
@inline flip_bits(state::UInt64,i::Int,j::Int)=state ⊻ (UInt64(1)<<i) ⊻ (UInt64(1)<<j)


# ------------------------
# Full Hilbert space H
function apply_H_full!(out::AbstractVector{T}, ψ::AbstractVector{T}, p::SpinParams) where T <: Number
    L = p.L
    N = length(ψ)
    @assert N == (1<<L)
    fill!(out, zero(T))

    nth = nthreads()
    outs = [zeros(T, N) for _ in 1:nth]

    hopping, onsite, zz = p.hopping_list, p.onsite_field, p.zz_list

    Threads.@threads for idx in 1:N
        tid = threadid()
        local_out = outs[tid]
        amp = ψ[idx]
        if amp == 0
            continue
        end
        state = UInt64(idx-1)
        diag = zero(T)
        # onsite field
        for i in 1:L
            diag += T(onsite[i]*sz_value(bit_at(state,i-1)))
        end
        # ZZ terms
        for (i,j,Jz) in zz
            diag += T(Jz*sz_value(bit_at(state,i-1))*sz_value(bit_at(state,j-1)))
        end
        @inbounds local_out[idx] += diag * amp

        # hopping terms
        for (i,j,Jxy) in hopping
            bi, bj = bit_at(state,i-1), bit_at(state,j-1)
            if bi != bj
                newstate = flip_bits(state,i-1,j-1)
                newidx = Int(newstate)+1
                @assert 1 <= newidx <= N "apply_H_full!: Index out of bounds: $newidx, L=$L"
                @inbounds local_out[newidx] += T(Jxy) * amp
            end
        end
    end

    for t in 1:nth
        @inbounds out .+= outs[t]
    end
    return out
end



# ------------------------
# Sector-restricted H
function apply_H_sector!(out::AbstractVector{T}, ψ::AbstractVector{T}, p::SpinParams,
                         states::Vector{UInt64}, idxmap::Dict{UInt64,Int}) where T <: Number
    L = p.L
    N = length(ψ)
    fill!(out, zero(T))
    nth = nthreads()
    outs = [zeros(T, N) for _ in 1:nth]

    hopping, onsite, zz = p.hopping_list, p.onsite_field, p.zz_list

    Threads.@threads for idx in 1:N
        tid = threadid()
        local_out = outs[tid]
        amp = ψ[idx]
        if amp == 0
            continue
        end
        state = states[idx]
        diag = zero(T)
        # diagonal terms
        for i in 1:L
            diag += T(onsite[i] * sz_value(bit_at(state,i-1)))
        end
        for (i,j,Jz) in zz
            diag += T(Jz * sz_value(bit_at(state,i-1)) * sz_value(bit_at(state,j-1)))
        end
        @inbounds local_out[idx] += diag * amp

        # off-diagonal hopping terms
        #Julia arrays are 1-based.
        # Bit positions in UInt64 are 0-based
        for (i,j,Jxy) in hopping
            bi, bj = bit_at(state,i-1), bit_at(state,j-1) # Bit positions in UInt64 are 0-based
            if bi != bj
                newstate = flip_bits(state,i-1,j-1)
                if haskey(idxmap, newstate)          # <-- safety check ,, to stay in same sector
                    newidx = idxmap[newstate] 
                    @inbounds local_out[newidx] += T(Jxy) * amp
                end
            end
        end
    end

    # combine threaded contributions
    for t in 1:nth
        @inbounds out .+= outs[t]
    end
    return out
end



end # module

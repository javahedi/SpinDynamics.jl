module Basis

    using Combinatorics

    
    export build_full_basis, build_sector_basis



    # Full basis: all 2^L states
    function build_full_basis(L::Int)
        N = 1 << L
        states = Vector{UInt64}(undef, N)
        idxmap = Dict{UInt64, Int}()
        @inbounds for i in 0:N-1
            s = UInt64(i)
            states[i+1] = s
            idxmap[s] = i+1
        end
        return states, idxmap
    end

    # Sector basis: all states with exactly nup up-spins
    function build_sector_basis(L::Int, nup::Int)
        states = UInt64[]
        sizehint!(states, binomial(L, nup))
        for comb in combinations(1:L, nup)
            s = UInt64(0)
            for i in comb
                s |= UInt64(1) << (i-1)
            end
            push!(states, s)
        end
        idxmap = Dict{UInt64, Int}()
        for (i, s) in enumerate(states)
            idxmap[s] = i
        end
        return states, idxmap
    end

    # bit access
    @inline function bit_at(state::UInt64, i::Int)
        return (state >> i) & 0x1
    end


end # module
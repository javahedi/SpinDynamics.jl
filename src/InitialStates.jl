
module InitialStates

export domain_wall_state_sector, domain_wall_state_full

# ---------------------------
"""
domain_wall_state(L, nup, states, idxmap)

Creates a domain-wall initial state in the given basis/sector.
- L: number of sites
- nup: number of up spins on the left side
- states: vector of basis bitstrings
- idxmap: mapping from bitstring to index

Returns:
- ψ0::Vector{Float64}: initial state vector
"""

function domain_wall_state_sector(L::Int, nup::Int, states::Vector{UInt64}, idxmap::Dict{UInt64,Int})
    s = UInt64(0)
    for i in 0:nup-1
        s |= UInt64(1)<<i
    end
    ψ0 = zeros(Float64,length(states))
    ψ0[idxmap[s]] = 1.0
    return ψ0
end


function domain_wall_state_full(L::Int)
    # nup = number of spins up on the left
    nup=Int(L/2)
    state = UInt64(0)
    for i in 0:(nup-1)
        state |= UInt64(1)<<i  # set leftmost nup bits to 1
    end
    return state
end


end # module


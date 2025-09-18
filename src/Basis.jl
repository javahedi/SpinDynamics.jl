module Basis

# Exported functions
export build_full_basis, build_sector_basis
using Combinatorics




"""
    build_full_basis(L::Int)

Constructs the full Hilbert space basis for L spins.
Returns:
- states::Vector{UInt64}: list of all basis states as bitstrings
- idxmap::Dict{UInt64,Int}: mapping from bitstring to index
"""
function build_full_basis(L::Int)
    N = 1 << L  # 2^L states
    states = [UInt64(i-1) for i in 1:N]
    idxmap = Dict(s => i for (i,s) in enumerate(states))
    return states, idxmap
end

"""
    build_sector_basis(L::Int, nup::Int)

Constructs the Hilbert space restricted to total Sz = nup - (L-nup)/2.
Returns:
- states::Vector{UInt64}: list of bitstrings in the sector
- idxmap::Dict{UInt64,Int}: mapping from bitstring to index
"""
function build_sector_basis(L::Int, nup::Int)
    states = UInt64[]
    for comb in combinations(0:L-1, nup)
        s = UInt64(0)
        for i in comb
            s |= UInt64(1)<<i
        end
        push!(states, s)
    end
    idxmap = Dict(s => i for (i,s) in enumerate(states))
    return states, idxmap
end

end # module

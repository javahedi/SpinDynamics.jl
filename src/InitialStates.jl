module InitialStates

export domain_wall_state_sector, domain_wall_state_full
export neel_state_sector, neel_state_full
export fm_state_sector, fm_state_full
export polarized_with_flips_sector, polarized_with_flips_full

# -------------------------------------------------
# Domain wall state
# -------------------------------------------------
function domain_wall_state_sector(L::Int, nup::Int, states::Vector{UInt64}, idxmap::Dict{UInt64,Int})
    s = UInt64(0)
    for i in 0:nup-1
        s |= UInt64(1) << i
    end
    ψ0 = zeros(Float64, length(states))
    ψ0[idxmap[s]] = 1.0
    return ψ0
end

function domain_wall_state_full(L::Int)
    nup = Int(ceil(L/2))
    s = UInt64(0)
    for i in 0:(nup-1)
        s |= UInt64(1) << i
    end
    return s
end

# -------------------------------------------------
# Néel state: ↑↓↑↓... (starting with ↑ at site 1)
# -------------------------------------------------
function neel_state_sector(L::Int, states::Vector{UInt64}, idxmap::Dict{UInt64,Int})
    s = UInt64(0)
    for i in 0:(L-1)
        if isodd(i+1)   # site index starting at 1
            s |= UInt64(1) << i
        end
    end
    ψ0 = zeros(Float64, length(states))
    ψ0[idxmap[s]] = 1.0
    return ψ0
end

function neel_state_full(L::Int)
    s = UInt64(0)
    for i in 0:(L-1)
        if isodd(i+1)
            s |= UInt64(1) << i
        end
    end
    return s
end

# -------------------------------------------------
# Ferromagnetic state: all ↑ or all ↓
# -------------------------------------------------
function fm_state_sector(L::Int, up::Bool, states::Vector{UInt64}, idxmap::Dict{UInt64,Int})
    s = up ? (UInt64(1) << L) - 1 : UInt64(0)
    ψ0 = zeros(Float64, length(states))
    ψ0[idxmap[s]] = 1.0
    return ψ0
end

function fm_state_full(L::Int, up::Bool)
    return up ? (UInt64(1) << L) - 1 : UInt64(0)
end

# -------------------------------------------------
# Polarized with flipped sites
# (start FM ↑, then flip given sites)
# -------------------------------------------------
function polarized_with_flips_sector(L::Int, flips::Vector{Int}, states::Vector{UInt64}, idxmap::Dict{UInt64,Int})
    s = (UInt64(1) << L) - 1  # all ↑
    for i in flips
        s ⊻= (UInt64(1) << (i-1))  # flip site i (1-based)
    end
    ψ0 = zeros(Float64, length(states))
    ψ0[idxmap[s]] = 1.0
    return ψ0
end

function polarized_with_flips_full(L::Int, flips::Vector{Int})
    s = (UInt64(1) << L) - 1
    for i in flips
        s ⊻= (UInt64(1) << (i-1))
    end
    return s
end

end # module

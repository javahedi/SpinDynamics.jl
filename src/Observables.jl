


module Observables

using FFTW
using Base.Threads
using ..Hamiltonian: SpinParams, sz_value, bit_at

export magnetization_per_site, structure_factor_Sq, connected_correlations
export magnetization_per_site_sector, structure_factor_Sq_sector, connected_correlations_sector

# --------------------------------------------------
# Magnetization per site (full Hilbert space)
# --------------------------------------------------
function magnetization_per_site(ψ::AbstractVector{T}, p::SpinParams) where {T<:Number}
    L, N = p.L, length(ψ)
    nth = nthreads()
    local_mags = [zeros(Float64, L) for _ in 1:nth]

    Threads.@threads for idx in 1:N
        prob = abs2(ψ[idx])
        if prob != 0.0
            state = UInt64(idx-1)
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
function connected_correlations(ψ::AbstractVector{T}, p::SpinParams) where {T<:Number}
    L, N = p.L, length(ψ)
    nth = nthreads()
    local_sz = [zeros(Float64, N, L) for _ in 1:nth]

    Threads.@threads for idx in 1:N
        prob = abs2(ψ[idx])
        if prob != 0.0
            state = UInt64(idx-1)
            tid = threadid()
            loc = local_sz[tid]
            for i in 0:(L-1)
                loc[idx,i+1] = sz_value(bit_at(state,i)) * prob
            end
        end
    end

    szvals = zeros(Float64, N, L)
    for loc in local_sz
        szvals .+= loc
    end

    S_i = sum(szvals, dims=1)[:]
    C_r = zeros(Float64,L)
    for r in 0:(L-1)
        tmp = 0.0
        for i in 1:L
            j = mod1(i+r,L)
            tmp += sum(szvals[:,i] .* szvals[:,j]) - S_i[i]*S_i[j]
        end
        C_r[r+1] = tmp / L
    end
    return C_r
end

# --------------------------------------------------
# Spin structure factor S(q)
# --------------------------------------------------
function structure_factor_Sq(ψ::AbstractVector{T}, p::SpinParams) where {T<:Number}
    C_r = connected_correlations(ψ, p)
    S_q = fft(C_r)
    qlist = [2π*(n-1)/p.L for n in 1:p.L]
    Sq_dict = Dict{Float64,Float64}()
    for n in 1:p.L
        Sq_dict[qlist[n]] = real(S_q[n])
    end
    return Sq_dict
end

# --------------------------------------------------
# Sector versions
# --------------------------------------------------
function magnetization_per_site_sector(ψ::AbstractVector{T}, p::SpinParams, states::Vector{UInt64}) where {T<:Number}
    L, N = p.L, length(ψ)
    nth = nthreads()
    local_mags = [zeros(Float64, L) for _ in 1:nth]

    Threads.@threads for idx in 1:N
        prob = abs2(ψ[idx])
        if prob != 0.0
            state = states[idx]
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

function connected_correlations_sector(ψ::AbstractVector{T}, p::SpinParams, states::Vector{UInt64}) where {T<:Number}
    L, N = p.L, length(ψ)
    nth = nthreads()
    local_sz = [zeros(Float64, N, L) for _ in 1:nth]

    Threads.@threads for idx in 1:N
        prob = abs2(ψ[idx])
        if prob != 0.0
            state = states[idx]
            tid = threadid()
            loc = local_sz[tid]
            for i in 0:(L-1)
                loc[idx,i+1] = sz_value(bit_at(state,i)) * prob
            end
        end
    end

    szvals = zeros(Float64, N, L)
    for loc in local_sz
        szvals .+= loc
    end

    S_i = sum(szvals, dims=1)[:]
    C_r = zeros(Float64,L)
    for r in 0:(L-1)
        tmp = 0.0
        for i in 1:L
            j = mod1(i+r,L)
            tmp += sum(szvals[:,i] .* szvals[:,j]) - S_i[i]*S_i[j]
        end
        C_r[r+1] = tmp / L
    end
    return C_r
end

function structure_factor_Sq_sector(ψ::AbstractVector{T}, p::SpinParams, states::Vector{UInt64}) where {T<:Number}
    C_r = connected_correlations_sector(ψ, p, states)
    S_q = fft(C_r)
    qlist = [2π*(n-1)/p.L for n in 1:p.L]
    Sq_dict = Dict{Float64,Float64}()
    for n in 1:p.L
        Sq_dict[qlist[n]] = real(S_q[n])
    end
    return Sq_dict
end

end # module


#=
module Observables

using FFTW

export magnetization_per_site, structure_factor_Sq, connected_correlations
export magnetization_per_site_sector, structure_factor_Sq_sector, connected_correlations_sector

using Base.Threads
using ..Hamiltonian: SpinParams, sz_value, bit_at


# --------------------------------------------------
# Helper functions (generic)
# --------------------------------------------------

# site-resolved magnetization
function magnetization_per_site(ψ::AbstractVector{T}, p::SpinParams) where T<:Number
    L = p.L
    N = length(ψ)
    mags = zeros(Float64, L)
    for idx in 1:N
        prob = abs2(ψ[idx])
        if prob == 0.0
            continue
        end
        state = UInt64(idx-1)
        for i in 0:(L-1)
            mags[i+1] += prob * sz_value(bit_at(state,i))
        end
    end
    return mags
end

# connected correlations C_r = <S_i S_{i+r}> - <S_i><S_{i+r}>
function connected_correlations(ψ::AbstractVector{T}, p::SpinParams) where T<:Number
    L = p.L
    N = length(ψ)
    sz_vals = zeros(Float64, N, L)
    for idx in 1:N
        prob = abs2(ψ[idx])
        if prob == 0.0
            continue
        end
        state = UInt64(idx-1)
        for i in 0:(L-1)
            sz_vals[idx,i+1] = sz_value(bit_at(state,i)) * prob
        end
    end
    S_i = sum(sz_vals, dims=1)[:]
    C_r = zeros(Float64, L)
    for r in 0:(L-1)
        tmp = 0.0
        for i in 1:L
            j = mod1(i+r, L)
            tmp += sum(sz_vals[:,i] .* sz_vals[:,j]) - S_i[i]*S_i[j]
        end
        C_r[r+1] = tmp / L
    end
    return C_r
end

# spin structure factor S(q)
function structure_factor_Sq(ψ::AbstractVector{T}, p::SpinParams) where T<:Number
    C_r = connected_correlations(ψ, p)
    S_q = fft(C_r)
    qlist = [2π*(n-1)/p.L for n in 1:p.L]
    Sq_dict = Dict{Float64, Float64}()
    for n in 1:p.L
        Sq_dict[qlist[n]] = real(S_q[n])
    end
    return Sq_dict
end

# --------------------------------------------------
# Observables per sector (generic)
# --------------------------------------------------

function magnetization_per_site_sector(ψ::AbstractVector{T}, p::SpinParams, states::Vector{UInt64}) where T<:Number
    L = p.L
    mags = zeros(Float64,L)
    for idx in 1:length(ψ)
        prob = abs2(ψ[idx])
        if prob == 0.0
            continue
        end
        state = states[idx]
        for i in 0:L-1
            mags[i+1] += prob * sz_value(bit_at(state,i))
        end
    end
    return mags
end

function connected_correlations_sector(ψ::AbstractVector{T}, p::SpinParams, states::Vector{UInt64}) where T<:Number
    L = p.L
    szvals = zeros(Float64, length(ψ), L)
    for idx in 1:length(ψ)
        prob = abs2(ψ[idx])
        if prob == 0.0
            continue
        end
        state = states[idx]
        for i in 0:L-1
            szvals[idx, i+1] = sz_value(bit_at(state,i)) * prob
        end
    end
    S_i = sum(szvals, dims=1)[:]
    C_r = zeros(Float64,L)
    for r in 0:L-1
        tmp = 0.0
        for i in 1:L
            j = mod1(i+r,L)
            tmp += sum(szvals[:,i] .* szvals[:,j]) - S_i[i]*S_i[j]
        end
        C_r[r+1] = tmp / L
    end
    return C_r
end

function structure_factor_Sq_sector(ψ::AbstractVector{T}, p::SpinParams, states::Vector{UInt64}) where T<:Number
    C_r = connected_correlations_sector(ψ, p, states)
    S_q = fft(C_r)
    qlist = [2π*(n-1)/p.L for n in 1:p.L]
    Sq_dict = Dict{Float64, Float64}()
    for n in 1:p.L
        Sq_dict[qlist[n]] = real(S_q[n])
    end
    return Sq_dict
end

end # module
=#
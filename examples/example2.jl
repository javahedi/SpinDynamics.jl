#!/usr/bin/env julia

using LinearAlgebra
using SpinDynamics
using Plots

# -----------------------
# System Setup
# -----------------------
L = 11
nup = 10
hopping = [(i, i+1, 1.0) for i in 1:(L-1)]  # J_xy = 1.0
zz = [(i, i+1, -1.0) for i in 1:(L-1)]      # J_z = 0
onsite = zeros(L)                           # h = 0
p = SpinParams(L, hopping, onsite, zz)

#@show p

states, idxmap = build_sector_basis(L, nup)  # Dimension = binomial(4,1) = 4


#@show states # [0x7, 0xb, 0xd, 0xe]  # Binary: 0111, 1011, 1101, 1110
#@show idxmap


middle_site = ceil(Int, L/2)
ψ0 = polarized_with_flips_sector(L, [middle_site], states, idxmap)
ψ0c = ComplexF64.(ψ0)

#=
N = length(ψ0)
for idx in 1:N
    state = UInt64(idx-1)
    for (i,j,Jxy) in hopping
        println("Bond: sites ($i, $j) -> bits ($(i-1), $(j-1))")
        bi, bj = bit_at(state,i-1), bit_at(state,j-1)
        println("  Bits: $bi, $bj")
        if bi != bj
            newstate = flip_bits(state,i-1,j-1)
            println("  Flipping bits $(i-1) and $(j-1)")
            println("  Old state: $(bitstring(state)[end-3:end])")
            println("  New state: $(bitstring(newstate)[end-3:end])")
        end
    end
end
=#

# -----------------------
# Exact Hamiltonian
# -----------------------
N = length(states)
H = zeros(ComplexF64, N, N)
for i in 1:N
    ψ_tmp = zeros(ComplexF64, N)
    ψ_tmp[i] = 1.0  # Set i-th basis state
    apply_H_sector!(H[:,i], ψ_tmp, p, states, idxmap)
end
H_ψ0 = H * ψ0c
# -----------------------
# Time Grid
# -----------------------
ts = range(0.0, 5.0, length=200)
mags_exact = Matrix{Float64}(undef, L, length(ts))
mags_cheb = Matrix{Float64}(undef, L, length(ts))
mags_krylov = Matrix{Float64}(undef, L, length(ts))
fidelities_cheb = Vector{Float64}(undef, length(ts))
fidelities_krylov = Vector{Float64}(undef, length(ts))

# -----------------------
# Energy Bounds
# -----------------------
Emin, Emax = estimate_energy_bounds(apply_H_sector!, ψ0c, p; m=30, states=states, idxmap=idxmap)
E_bounds = (Emin - 1e-8, Emax + 1e-8)
@show E_bounds
# -----------------------
# Time Evolution
# -----------------------
@time for (j, t) in enumerate(ts)
    # Exact
    ψt_exact = exp(-im * t * H) * ψ0c
    ψt_exact ./= norm(ψt_exact)
    mags_exact[:,j] = magnetization_per_site_sector(ψt_exact, p, states)

    # Chebyshev
    ψt_cheb = chebyshev_time_evolve(ψ0c, t, apply_H_sector!, p; M=200, E_bounds=E_bounds, states=states, idxmap=idxmap)
    mags_cheb[:,j] = magnetization_per_site_sector(ψt_cheb, p, states)
    fidelities_cheb[j] = abs2(dot(ψt_exact, ψt_cheb))

    # Krylov
    ψt_krylov = krylov_time_evolve(ψ0c, t, apply_H_sector!, p; m=50, states=states, idxmap=idxmap)
    mags_krylov[:,j] = magnetization_per_site_sector(ψt_krylov, p, states)
    fidelities_krylov[j] = abs2(dot(ψt_exact, ψt_krylov))
end

# -----------------------
# Plotting
# -----------------------
plt1 = heatmap(1:L, ts, -1.0 .* mags_exact', xlabel="Site", ylabel="Time", title="Exact", colorbar_title="⟨Sz⟩")
plt2 = heatmap(1:L, ts, -1.0 .* mags_cheb', xlabel="Site", ylabel="Time", title="Chebyshev", colorbar_title="⟨Sz⟩")
plt3 = heatmap(1:L, ts, -1.0 .* mags_krylov', xlabel="Site", ylabel="Time", title="Krylov", colorbar_title="⟨Sz⟩")
plt4 = plot(ts, [fidelities_cheb, fidelities_krylov], label=["Chebyshev" "Krylov"], xlabel="Time", ylabel="Fidelity |⟨ψ_exact|ψ⟩|²", title="Fidelity vs Exact")

plot(plt1, plt2, plt3, plt4, layout=(2,2), size=(1200,800))
savefig("examples/magnetization_comparison_L$(L)_nup$(nup).png")

println("Saved plots to magnetization_comparison_L$(L)_nup$(nup).png")
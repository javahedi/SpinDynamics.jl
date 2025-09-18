#!/usr/bin/env julia

using SpinDynamics
using LinearAlgebra
using Plots

# -------------------------
# System parameters
# -------------------------
L = 21
nup = L - 1
hopping = [(i, i+1, 1.0) for i in 1:(L-1)]
zz      = [(i, i+1, 0.0) for i in 1:(L-1)]
onsite  = zeros(L)
p = SpinParams(L, hopping, onsite, zz)

# -------------------------
# Basis and initial state
# -------------------------
states, idxmap = build_sector_basis(L, nup) # states = [UInt64(1), UInt64(2), UInt64(4)]  # |001⟩, |010⟩, |100⟩

#ψ0 = neel_state_sector(L, states, idxmap)              # Néel ↑↓↑↓...
#ψ0 = fm_state_sector(L, true, states, idxmap)          # all ↑
#ψ0 = fm_state_sector(L, false, states, idxmap)         # all ↓
middle_site = ceil(Int, L/2)
ψ0 = polarized_with_flips_sector(L, [middle_site], states, idxmap) # all ↑ but flip middle site
#ψ0 = polarized_with_flips_sector(L, [3,7], states, idxmap)  # flip sites 3 and 7
#ψ0 = domain_wall_state_sector(L, nup, states, idxmap) 

#state_dw = domain_wall_state_full(L)
#idx_dw = Int(state_dw) + 1   # because you index states 0-based -> 1-based Julia array
#ψ0 = zeros(Float64, 1 << L )
#ψ0[idx_dw] = 1.0 

ψ0c = ComplexF64.(ψ0)
@show magnetization_per_site_sector(ψ0c, p, states)
#@show magnetization_per_site(ψ0c, p)


# -------------------------
# Time grid (log scale)
# -------------------------
#ts = exp.(range(log(0.01), log(100.0), length=120))  # finer resolution
ts = range(0.0, 5.0; length=200)   # 100 points between 0 and 20

# -------------------------
# Krylov evolution
# -------------------------
mags_krylov = Matrix{Float64}(undef, L, length(ts))
@time for (j, t) in enumerate(ts)
    ψt = krylov_time_evolve(ψ0c, t, apply_H_sector!, p;
                            m=30, states=states, idxmap=idxmap)
    mags_krylov[:, j] .= magnetization_per_site_sector(ψt, p, states)
    

    #ψt = krylov_time_evolve(ψ0c, t, apply_H_full!, p; m=30)
    #mags_krylov[:, j] .= magnetization_per_site(ψt, p)

end

# -------------------------
# Chebyshev evolution
# -------------------------

# Energy bounds
Emin, Emax = estimate_energy_bounds(apply_H_sector!, ψ0c, p; 
                                    m=80, states=states, idxmap=idxmap)
#Emin, Emax = estimate_energy_bounds(apply_H_full!, ψ0c, p; m=80)
E_bounds = (Emin - 1e-6, Emax + 1e-6)
#@show E_bounds

# Time evolution
mags_cheb = Matrix{Float64}(undef, L, length(ts))

@time for (j, t) in enumerate(ts)
    #ψt = chebyshev_time_evolve(ψ0c, t, apply_H_full!, p;
    #                           M=500, E_bounds=E_bounds)
    #mags_cheb[:, j] .= magnetization_per_site(ψt, p)

    ψt = chebyshev_time_evolve(ψ0c, t, apply_H_sector!, p;
                              M=500, E_bounds=E_bounds,
                              states=states, idxmap=idxmap)
    mags_cheb[:, j] .= magnetization_per_site_sector(ψt, p, states)
end

# -------------------------
# Plot heatmaps
# -------------------------

plt1 = heatmap(1:L, ts, -1.0.*mags_krylov';
               xlabel="site", ylabel="time", 
               title="Krylov", colorbar_title="⟨Sz⟩")

plt2 = heatmap(1:L, ts, -1.0.*mags_cheb';
               xlabel="site", ylabel="time", 
               title="Chebyshev", colorbar_title="⟨Sz⟩")

plot(plt1, plt2, layout=(1,2), size=(1000,400))
savefig("examples/magnetization_mesh_L$(L)_nup$(nup)_timeY.png")

println("Saved mesh plots to magnetization_mesh_L$(L)_nup$(nup)_timeY.pdf")

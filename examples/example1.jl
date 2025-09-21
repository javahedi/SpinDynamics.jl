#!/usr/bin/env julia

using SpinDynamics
using LinearAlgebra
using Plots

# -------------------------
# System parameters
# -------------------------
L = 20
nup = 10 #L - 1
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
#middle_site = ceil(Int, L/2)
#ψ0 = polarized_with_flips_sector(L, [middle_site], states, idxmap) # all ↑ but flip middle site
#ψ0 = polarized_with_flips_sector(L, [3,7], states, idxmap)  # flip sites 3 and 7
ψ0 = domain_wall_state_sector(L, nup, states, idxmap) 

#state_dw = domain_wall_state_full(L)
#idx_dw = Int(state_dw) + 1   # because you index states 0-based -> 1-based Julia array
#ψ0 = zeros(Float64, 1 << L )
#ψ0[idx_dw] = 1.0


N = length(ψ0)
ψ0c = ComplexF64.(ψ0)
@show magnetization_per_site_sector(ψ0c, p, states)
#@show magnetization_per_site(ψ0c_kry, p)


# -------------------------
# Time grid (log scale)
# -------------------------
#ts = exp.(range(log(0.01), log(100.0), length=120))  # finer resolution
#ts = 10 .^ range(log10(0.01), log10(5.0); length=200)
ts = range(0.0, 5.0; length=100)   # 200 points between 0 and 5

# -------------------------
# Krylov evolution
# -------------------------

# Initialize magnetization array
mags_krylov = Matrix{Float64}(undef, L, length(ts))


"""
m=30
n = length(ψ0c)
ws = KrylovWorkspace(n, m)

# initial observable
mags_krylov[:, 1] .= magnetization_per_site_sector(ψ0c, p, states)

ψ_tmp = similar(ψ0c)
@time for j in 1:length(ts)-1
    dt = ts[j+1] - ts[j]
    #global ψ_tmp
    krylov_time_evolve!(ψ_tmp, ψ0c, dt, apply_H_sector!, p, ws;
                        m=m, states=states, idxmap=idxmap)
    copyto!(ψ0c, ψ_tmp)
    mags_krylov[:, j+1] .= magnetization_per_site_sector(ψ0c, p, states)
end



ψt    = copy(ψ0c)                                      
# Loop over consecutive time steps
@time for j in 1:length(ts)-1
    dt = ts[j+1] - ts[j]
    global ψt  # Add this line to explicitly declare it as global
    ψt = krylov_time_evolve(ψt, dt, apply_H_sector!, p; 
                                m=30, states=states, idxmap=idxmap)
    mags_krylov[:, j] .= magnetization_per_site_sector(ψt, p, states)

end
"""

# fast ,, but i don't knwo why it is faster than above!!!
@time for (j, t) in enumerate(ts)
    ψtt = krylov_time_evolve(ψ0c, t, apply_H_sector!, p; 
                                m=30, states=states, idxmap=idxmap)

    mags_krylov[:, j] .= magnetization_per_site_sector(ψtt, p, states)

end


"""
# -------------------------
# Chebyshev evolution
# -------------------------
# Time evolution
mags_cheb = Matrix{Float64}(undef, L, length(ts))
Emin, Emax = estimate_energy_bounds(apply_H_sector!, ψ0, p; 
                                        m=80, states=states, idxmap=idxmap)
                                               
ψ_tmp = copy(ψ0c)
mags_cheb[:, 1] .= magnetization_per_site_sector(ψ0c, p, states)
@time  for j in 1:length(ts)-1 
    dt = ts[j+1] - ts[j]
    global ψ_tmp
    ψ_tmp = chebyshev_time_evolve(ψ_tmp, dt, apply_H_sector!, p;
                                  n=10, Ebounds=(Emin, Emax), 
                                  states=states, idxmap=idxmap)
    mags_cheb[:, j+1] .= magnetization_per_site_sector(ψ_tmp, p, states)

    #ψ_tmp = chebyshev_time_evolve(ψ_tmp, t, apply_H_full!, p; n=50)
    #mags_cheb[:, j] .= magnetization_per_site(ψ_tmp, p)    
end
"""

# -------------------------
# Chebyshev evolution
# -------------------------

# Time evolution
mags_cheb = Matrix{Float64}(undef, L, length(ts))

# Estimate energy bounds (using the new function signature)
Emin, Emax = estimate_energy_bounds(apply_H_sector!, ψ0c, p; 
                                    m=80, states=states, idxmap=idxmap)

# Create a workspace for repeated time evolution to avoid allocations
workspace = ChebyshevWorkspace(ψ0c)

# Initialize with first time point
mags_cheb[:, 1] .= magnetization_per_site_sector(ψ0c, p, states)

ψ_tmp = copy(ψ0c)

@time for j in 1:length(ts)-1 
    dt = ts[j+1] - ts[j]
    global ψ_tmp
    # Use the workspace for efficient repeated evolution
    ψ_tmp = chebyshev_time_evolve(ψ_tmp, dt, apply_H_sector!, p;
                                  n=10, Ebounds=(Emin, Emax), 
                                  states=states, idxmap=idxmap,
                                  workspace=workspace)
    
    mags_cheb[:, j+1] .= magnetization_per_site_sector(ψ_tmp, p, states)
end
#---------------------
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

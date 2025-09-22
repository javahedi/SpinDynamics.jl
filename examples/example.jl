#!/usr/bin/env julia

using LinearAlgebra
using SpinDynamics
using Plots

# -----------------------
# System Setup
# -----------------------
L = 14
nup = 7


model_nn = build_model(L; nup=nup, hopping=nn_hopping(L,1.0), onsite_field=zeros(L), zz=nn_hopping(L,0.5))
#Model_lr = build_model(L; nup=nup, hopping=long_range_hopping(L,(i,j)->1.0/(abs(i-j)^3)), onsite_field=zeros(L), zz=[])


middle_site = ceil(Int, L/2)
#ψ0 = polarized_state_with_flips(model_nn, [1,L])
#ψ0 = domain_wall_state(model_nn)
ψ0 = neel_state(model_nn)
ψ0c = ComplexF64.(ψ0)



# -----------------------
# Exact Hamiltonian
# -----------------------
N = length(model_nn.states)
H = zeros(ComplexF64, N, N)
for i in 1:N
    ψ_tmp = zeros(ComplexF64, N)
    ψ_tmp[i] = 1.0  # Set i-th basis state
    apply_H!(H[:,i], ψ_tmp, model_nn)
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



Emin, Emax = estimate_energy_bounds(apply_H!, length(ψ0c), model_nn; lanc_m=80)

#=----------------------
@time for (j, t) in enumerate(ts)
    # Exact
    ψt_exact = exp(-im * t * H) * ψ0c
    ψt_exact ./= norm(ψt_exact)
    mags_exact[:,j] = magnetization_per_site(ψt_exact, model_nn)


    # Time evolution
    ψt = chebyshev_time_evolve(ψ0c, t, apply_H!, model_nn, Ebounds=(Emin, Emax), cheb_n=15)
    mags_cheb[:,j] = magnetization_per_site(ψt, model_nn)
    fidelities_cheb[j] = abs2(dot(ψt_exact, ψt))

    # Time evolution
    ψt = krylov_time_evolve(ψ0c, t, apply_H!, model_nn, kry_m=50)
    mags_krylov[:,j] = magnetization_per_site(ψt, model_nn)
    fidelities_krylov[j] = abs2(dot(ψt_exact, ψt))

end
=#

ψt_exact = copy(ψ0c)
ψt_cheb = copy(ψ0c)
ψt_krylov = copy(ψ0c)   
mags_exact[:,1] = magnetization_per_site(ψt_exact, model_nn)
mags_cheb[:,1] = mags_exact[:,1]
mags_krylov[:,1] = mags_exact[:,1]
fidelities_cheb[1] = 1.0
fidelities_krylov[1] = 1.0 

dt = ts[2] - ts[1]
expH = exp(-im * dt * H)
@time for  j in 1:length(ts)-1
    global ψt_exact, ψt_cheb, ψt_krylov
    dt = ts[j+1] - ts[j]
    # Exact
    ψt_exact = expH * ψt_exact
    ψt_exact ./= norm(ψt_exact)
    mags_exact[:,j+1] = magnetization_per_site(ψt_exact, model_nn)


    # Time evolution
    ψt_cheb = chebyshev_time_evolve(ψt_cheb, dt, apply_H!, model_nn, Ebounds=(Emin, Emax), cheb_n=10)
    mags_cheb[:,j+1] = magnetization_per_site(ψt_cheb, model_nn)
    fidelities_cheb[j+1] = abs2(dot(ψt_exact, ψt_cheb))

    # Time evolution
    ψt_krylov = krylov_time_evolve(ψt_krylov, dt, apply_H!, model_nn, kry_m=50)
    mags_krylov[:,j+1] = magnetization_per_site(ψt_krylov, model_nn)
    fidelities_krylov[j+1] = abs2(dot(ψt_exact, ψt_krylov))

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
savefig("examples/magnetization_comparison_L$(L)_nup$(nup).pdf")

println("Saved plots to magnetization_comparison_L$(L)_nup$(nup).png")

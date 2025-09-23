#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays
using ArnoldiMethod   # from ArnoldiIteration.jl
using SpinDynamics

# -----------------------
# Parameters
# -----------------------
L = 8
nup = div(L, 2)  # Sz = 0 sector
Jxy = 1.0
Jz  = 0.5
ω_range = range(-5.0, 5.0, length=200)

# -----------------------
# Build sector model
# -----------------------
model = build_model(L; nup=nup,
                    hopping=nn_hopping(L, Jxy),
                    onsite_field=zeros(L),
                    zz=nn_hopping(L, Jz))

# -----------------------
# Construct exact Hamiltonian (sparse)
# -----------------------
N = length(model.states)
rows = Int[]
cols = Int[]
vals = Float64[]

# build H by applying to basis states
for j in 1:N
    ψ = zeros(Float64, N)
    ψ[j] = 1.0
    out = zeros(Float64, N)
    apply_H!(out, ψ, model)
    for i in 1:N
        if abs(out[i]) > 1e-12
            push!(rows, i)
            push!(cols, j)
            push!(vals, out[i])
        end
    end
end
H = sparse(rows, cols, vals, N, N)
println("Hamiltonian dimension = $N × $N with $(nnz(H)) nonzeros.")

# -----------------------
# Find ground state with Arnoldi (Lanczos)
# -----------------------
nev = 1      # number of eigenpairs
which = :SR  # smallest real
decomp, history = partialschur(H; nev=1, tol=1e-5, restarts=100, which=:SR)

Emin , ψ0= partialeigen(decomp)
ψ0 = Array(decomp.Q)[:,1]            # ground state eigenvector
@show norm(ψ0), typeof(ψ0)
ψ0c = ComplexF64.(ψ0)
ψ0c ./= norm(ψ0c)
println("Ground state energy = $Emin")

decomp, history = partialschur(H; nev=1, tol=1e-5, restarts=100, which=:LR)
Emax , _= partialeigen(decomp)

println("Maximum state energy = $Emax")
# -----------------------
# Local autocorrelation spectrum C_ii(ω)
# -----------------------
site = div(L, 2)
opSz = create_spin_operator(site, :z)
Cii = kpm_dynamical_correlation(ψ0c, opSz, opSz, ω_range, apply_H!, model; n=50)
@show Cii
# -----------------------
# Density of states (average over sites)
# -----------------------
ρ = zeros(length(ω_range))
for i in 1:L
    opSz_i = create_spin_operator(i, :z)
    ρ .+= kpm_dynamical_correlation(ψ0c, opSz_i, opSz_i, ω_range, 
                                    apply_H!, model; n=50)
end
ρ ./= L

#@show ρ
#=
# -----------------------
# Dynamical structure factor S(q, ω)
# -----------------------
C = kpm_correlation_matrix(ComplexF64.(ψ0), ω_range, apply_H!, model; n=300,
                               opA_type_a=:z, opB_type_b=:z)
positions = collect(1:L)
qs = range(0, π, length=50)

#Sqω(C::Array{Float64,3}, q::Float64, positions::Vector{Float64})
Sqω_ = [Sqω(C, q, Float64.(positions)) for q in qs]


# -----------------------
# Plotting
# -----------------------
using Plots

plt1 = plot(ω_range, Cii, xlabel="ω", ylabel="Cii(ω)",
            title="Local spectral function, site=$site")

plt2 = plot(ω_range, ρ, xlabel="ω", ylabel="ρ(ω)",
            title="Spin spectral density (DOS)")

plt3 = heatmap(qs, ω_range, hcat(Sqω_...), xlabel="q", ylabel="ω",
               title="Dynamical structure factor S(q,ω)",
               colorbar_title="Intensity")

plot(plt1, plt2, plt3, layout=(2,2), size=(1200,800))
savefig("examples/kpm_xxz_spectra_L$(L)_Sz0.pdf")

println("Saved spectra plots to examples/kpm_xxz_spectra_L$(L)_Sz0.pdf")
=#
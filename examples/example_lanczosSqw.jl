#!/usr/bin/env julia

using SpinDynamics
using Plots
using LaTeXStrings

# ------------------------------------------------------------
# Dynamical structure factor S(q, ω) with Lanczos
#
# Run from the repository root:
#   julia --project=. examples/example_lanczosSqw.jl
# ------------------------------------------------------------

L = 16

model = XXZChain(
    L;
    Jxy = 1.0,
    Jz = 1.0,
    nup = L ÷ 2,
)

println("Hilbert-space dimension: ", length(model.states))

# Ground state
@time E0, ψ0 = groundstate(
    model;
    lanc_m = 100,
)

println("Ground-state energy: ", E0)

# Momentum and frequency grids
q = momenta(model)
ω = collect(range(0.0, 5.0; length=100))

# S(q, ω)
@time S = dynamical_structure_factor(
    model,
    ψ0,
    q,
    ω;
    method = :lanczos,
    lanc_m = 100,
    eta = 0.05,
)

# Plot
plt = heatmap(
    q,
    ω,
    S';
    xlabel = L"q",
    ylabel = L"\omega",
    title = L"S^z(q,\omega)\ \mathrm{--\ Lanczos}",
    colorbar_title = "S",
    aspect_ratio = :auto,
    xticks = (
        [0, π / 2, π, 3π / 2],
        [L"0", L"\pi/2", L"\pi", L"3\pi/2"],
    ),
)

outfile = joinpath(
    @__DIR__,
    "lanczos_xxz_spectra_L$(L)_Sz0.png",
)

savefig(plt, outfile)

println("Saved figure to: ", outfile)



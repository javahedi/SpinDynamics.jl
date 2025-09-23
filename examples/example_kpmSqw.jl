using LinearAlgebra
using SparseArrays
using ArnoldiMethod   # from ArnoldiIteration.jl
using SpinDynamics
using Plots
using LaTeXStrings  # for pretty labels

# -----------------------
# Parameters
# -----------------------
L = 20
nup = div(L, 2)  # Sz = 0 sector
Jxy = 1.0
Jz  = 1.0
ω_range = range(0.0, 5.0, length=50)
ω_range=collect(ω_range)

# -----------------------
# Build sector model
# -----------------------
model = build_model(L; nup=nup,
                    hopping=nn_hopping(L, Jxy),
                    onsite_field=zeros(L),
                    zz=nn_hopping(L, Jz))




@time Emin_mein , ψ0_mein = lanczos_groundstate(apply_H!, model)
@show norm(ψ0_mein), typeof(ψ0_mein), Emin_mein


E_min, E_max = estimate_energy_bounds(apply_H!, model; lanc_m=100)
a = (E_max - E_min) / 2 * 0.95
b = (E_max + E_min) / 2
@show E_min, E_max
@show a, b

#---------------------------------------
q_list =  collect(2π * (0:model.L-1) / model.L)
#q_list =  collect(range(0, 2π, length=64))


@time Smat = kpm_sqw(ComplexF64.(ψ0_mein), model, q_list, ω_range; 
                                    a=a, b=b, kpm_m=100, kernel=:jackson)

#@show Smat



heatmap(q_list, ω_range, Smat',
    xlabel = L"q",
    ylabel = L"\omega",
    title = L"$S^z(q,\omega)~{\rm KPM method}$",
    colorbar_title = "S",
    aspect_ratio = :auto,
    #clims = (0.0, 1.0),
    xticks = ([0, π/2, π, 3π/2, 2π],
              [L"0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"])
)


savefig("examples/kpm_xxz_spectra_L$(L)_Sz0.png")
println("Saved spectra plots to examples/kpm_xxz_spectra_L$(L)_Sz0.png")

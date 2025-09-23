using LinearAlgebra
using SparseArrays
using ArnoldiMethod   # from ArnoldiIteration.jl
using SpinDynamics
using Plots
using LaTeXStrings  # for pretty labels

# -----------------------
# Parameters
# -----------------------
L = 16
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




#=
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
@time decomp, history = partialschur(H; nev=1, tol=1e-5, restarts=100, which=:SR)

Emin , ψ0= partialeigen(decomp)
ψ0 = Array(decomp.Q)[:,1]            # ground state eigenvector

@show norm(ψ0), typeof(ψ0), Emin
=#


@time Emin_mein , ψ0_mein = lanczos_groundstate(apply_H!, model)
@show norm(ψ0_mein), typeof(ψ0_mein), Emin_mein


#=
# Compare with Arnoldi
final_diff = norm(ψ0 - ψ0_mein)
final_overlap = abs(dot(ψ0, ψ0_mein))

println("Final results:")
println("Energy difference: ", abs(Emin[1] - Emin_mein))
println("State norm difference: ", final_diff)
println("State overlap: ", final_overlap)
=#


#---------------------------------------
q_list =  collect(2π * (0:model.L-1) / model.L)
#q_list =  collect(range(0, 2π, length=64))


@time Smat = lanczos_sqw(ψ0_mein, model, q_list, ω_range; lanc_m=100, eta=0.05)



@show Smat

heatmap(q_list, ω_range, Smat',
    xlabel = L"q",
    ylabel = L"\omega",
    title = L"$S^z(q,\omega)~{\rm Lanczos method}$",
    colorbar_title = "S",
    aspect_ratio = :auto,
    xticks = ([0, π/2, π, 3π/2, 2π],
              [L"0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"])
)


savefig("examples/lanczos_xxz_spectra_L$(L)_Sz0.png")
println("Saved spectra plots to examples/lanczos_xxz_spectra_L$(L)_Sz0.png")

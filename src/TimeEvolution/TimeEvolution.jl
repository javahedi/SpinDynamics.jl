module TimeEvolution



# Re_Export
export chebyshev_time_evolve, lanczos_extremal, estimate_energy_bounds, 
        run_chebyshev, ChebyshevWorkspace, apply_rescaled_H!


export krylov_time_evolve!, krylov_time_evolve, KrylovWorkspace,
         run_krylov, KrylovWorkspace

export kpm_dynamical_correlation, compute_chebyshev_moments,
         get_jackson_kernel, evaluate_chebyshev_series,
        run_kpm_dynamical




# Include the engines
include("Chebyshev.jl")
include("Krylov.jl")
#include("QuantumTypicality.jl")
include("KPM.jl")

# Make submodules available
using .Krylov
using .Chebyshev
using .KPM
#using .QuantumTypicality

end # module

module TimeEvolution



# Re_Export
export chebyshev_time_evolve, run_chebyshev, 
        ChebyshevWorkspace


export krylov_time_evolve!, krylov_time_evolve, 
        KrylovWorkspace, run_krylov, KrylovWorkspace

export kpm_dynamical_correlation, compute_chebyshev_moments,
        get_jackson_kernel, evaluate_chebyshev_series,
        run_kpm_dynamical, kpm_correlation_matrix, Sqω




# Include the engines
include("Chebyshev.jl")
include("Krylov.jl")
include("KPM.jl")
#include("QuantumTypicality.jl")


# Make submodules available
using .Krylov
using .Chebyshev
using .KPM
#using .QuantumTypicality

end # module

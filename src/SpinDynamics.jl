module SpinDynamics

# Re_Export top-level things
export  SpinParams, bit_at, sz_value, flip_bits
export domain_wall_state_sector, domain_wall_state_full 
export apply_H_full!, apply_H_sector!
export build_sector_basis, build_full_basis
export magnetization_per_site, structure_factor_Sq, connected_correlations
export magnetization_per_site_sector, structure_factor_Sq_sector, connected_correlations_sector

export chebyshev_time_evolve, lanczos_extremal
export run_chebyshev_sector, run_chebyshev_full


export krylov_time_evolve_sector, run_krylov_sector
export krylov_time_evolve_full, run_krylov_full



export TimeEvolution  # <--- make submodule visible

# Include modules
include("Basis.jl")
include("Hamiltonian.jl")
include("InitialStates.jl")
include("Observables.jl")
include("TimeEvolution/TimeEvolution.jl")

using .Basis
using .Hamiltonian
using .InitialStates
using .Observables
using .TimeEvolution

end

module SpinDynamics



    include("Basis.jl")
    using .Basis
    export build_full_basis, build_sector_basis


    include("SpinModel.jl")
    using .SpinModel
    export Model, build_model, nn_hopping, long_range_hopping

    include("Hamiltonian.jl")
    using .Hamiltonian
    export apply_H!, apply_rescaled_H!, 
           bit_at, sz_value, flip_bits, create_spin_operator, Sz_q_vector

    include("Lanczos.jl")
    using .Lanczos
    export lanczos_extremal, lanczos_groundstate, 
           lanczos_tridiag, estimate_energy_bounds

    
  

    include("InitialStates.jl")
    using .InitialStates
    export domain_wall_state, neel_state, polarized_state, polarized_state_with_flips
            


    include("Observables.jl")
    using .Observables
    export magnetization_per_site, structure_factor_Sq, connected_correlations
            


    include("LanczosSqw.jl")
    using .LanczosSqw
    export lanczos_sqw

    include("KPM_Sqw.jl")
    using .KPM_Sqw
    export kpm_sqw

    


    include("TimeEvolution/TimeEvolution.jl")
    using .TimeEvolution
    export TimeEvolution 


    export chebyshev_time_evolve, estimate_energy_bounds,
            run_chebyshev, ChebyshevWorkspace

    export krylov_time_evolve!, krylov_time_evolve, KrylovWorkspace, run_krylov

    export kpm_dynamical_correlation, kpm_dynamical_correlation_matrix, 
            run_kpm_dynamical, kpm_correlation_matrix, Sqω


end

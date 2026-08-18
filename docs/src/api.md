# API

This page lists the main SpinDynamics.jl interfaces and selected lower-level routines.

```@meta
CurrentModule = SpinDynamics
```

## Model construction

```@docs
XXZChain
build_model
momenta
```

## Ground states and Lanczos

```@docs
groundstate
lanczos_extremal
lanczos_tridiag
estimate_energy_bounds
```

## Time evolution

```@docs
time_evolve
krylov_time_evolve
krylov_time_evolve!
chebyshev_time_evolve
```

## Observables and operators

```@docs
structure_factor
create_spin_operator
apply_rescaled_H!
```

## Dynamical structure factors

```@docs
dynamical_structure_factor
```
# Internals

This page documents lower-level implementation helpers used internally by SpinDynamics.jl.

```@meta
CurrentModule = SpinDynamics
```

## Time-evolution helpers

```@docs
SpinDynamics.TimeEvolution.Chebyshev.run_chebyshev
SpinDynamics.TimeEvolution.Krylov.run_krylov
```

## KPM time-evolution helpers

```@docs
SpinDynamics.TimeEvolution.KPM.get_rescaling_params
SpinDynamics.TimeEvolution.KPM.get_jackson_kernel
SpinDynamics.TimeEvolution.KPM.evaluate_chebyshev_series
SpinDynamics.TimeEvolution.KPM.compute_cross_chebyshev_moments
SpinDynamics.TimeEvolution.KPM.kpm_dynamical_correlation
```

## KPM spectroscopy helpers

```@docs
SpinDynamics.KPM_Sqw.get_rescaling_params
```
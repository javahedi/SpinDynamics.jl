# SpinDynamics.jl

SpinDynamics.jl is a Julia package for matrix-free simulations of quantum spin systems.

It provides tools for constructing spin models, computing ground states, performing real-time evolution, and calculating static and dynamical spin structure factors.

## Features

- XXZ spin-chain models
- Matrix-free Hamiltonian application
- Lanczos ground-state calculations
- Krylov time evolution
- Chebyshev time evolution
- Static structure factors
- Dynamical structure factors using Lanczos
- Dynamical structure factors using the Kernel Polynomial Method (KPM)

## Quick start

```julia
using SpinDynamics

L = 16

model = XXZChain(
    L;
    Jxy = 1.0,
    Jz = 1.0,
    nup = div(L, 2),
)

E0, ψ0 = groundstate(model)

println("Ground-state energy: ", E0)
```

## Documentation

```@contents
Pages = [
    "time_evolution.md",
    "spectroscopy.md",
    "api.md",
]
Depth = 2
```
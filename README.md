# SpinDynamics.jl

[![Build Status](https://github.com/javahedi/SpinDynamics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/javahedi/SpinDynamics.jl/actions/workflows/CI.yml?query=branch%3Amain)

<p align="center">
  <img src="docs/src/assets/lanczos_xxz_spectra_L20_Sz0.png"
       alt="Lanczos dynamical structure factor for the XXZ chain"
       width="720">
</p>

**SpinDynamics.jl** is a Julia package for ground-state calculations, real-time evolution, and dynamical spectroscopy of quantum spin systems.

It provides matrix-free Lanczos, Krylov, Chebyshev, and Kernel Polynomial Method (KPM) algorithms together with a compact high-level API for common spin-1/2 workflows.

> **Status:** SpinDynamics.jl is under active development. The API is usable but may evolve as the package develops.

## Features

- Spin-1/2 models in the full Hilbert space or fixed-`nup` sectors
- Matrix-free Hamiltonian application
- Lanczos ground-state calculations
- Krylov and Chebyshev real-time evolution
- Static spin structure factors
- Dynamical structure factors `S(q, ω)` using Lanczos or KPM
- Common initial states and local observables

## Installation

SpinDynamics.jl supports **Julia 1.10 and later**.

Until registration in the Julia General registry, install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/javahedi/SpinDynamics.jl")
```

## Quick start

```julia
using SpinDynamics

L = 16

model = XXZChain(
    L;
    Jxy = 1.0,
    Jz = 1.0,
    nup = L ÷ 2,
)

E0, ψ0 = groundstate(model)

q = momenta(model)
ω = range(0.0, 5.0; length=100)

S = dynamical_structure_factor(
    model,
    ψ0,
    q,
    ω;
    method = :lanczos,
    lanc_m = 100,
    eta = 0.05,
)
```

Real-time evolution uses the same high-level interface:

```julia
ψt = time_evolve(model, ψ0, 0.5; method=:krylov)
```

## Documentation

The documentation contains usage examples, numerical-method descriptions, spectroscopy examples, time-evolution comparisons, and the API reference.

- [Documentation](https://javahedi.github.io/SpinDynamics.jl/)
- [Time evolution](https://javahedi.github.io/SpinDynamics.jl/dev/time_evolution/)
- [Spectroscopy](https://javahedi.github.io/SpinDynamics.jl/dev/spectroscopy/)
- [API reference](https://javahedi.github.io/SpinDynamics.jl/dev/api/)

## Examples

Runnable scripts are kept in [`examples/`](examples/):

- [`example_lanczosSqw.jl`](examples/example_lanczosSqw.jl)
- [`example_kpmSqw.jl`](examples/example_kpmSqw.jl)
- [`example_time_evolution.jl`](examples/example_time_evolution.jl)

The plotting dependencies used by these scripts are isolated in `examples/Project.toml`.

## Public API

| Task | Function |
| --- | --- |
| XXZ model construction | `XXZChain` |
| Momentum grid | `momenta` |
| Ground state | `groundstate` |
| Real-time evolution | `time_evolve` |
| Static structure factor | `structure_factor` |
| Dynamical structure factor | `dynamical_structure_factor` |

Lower-level Lanczos, KPM, Krylov, Chebyshev, basis, Hamiltonian, initial-state, and observable routines are also available for advanced use.

## Testing

Run the complete test suite with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test suite includes package-quality checks with Aqua.jl and is also run through GitHub Actions.

## Contributing

Contributions, bug reports, and suggestions are welcome through GitHub issues and pull requests.

## License

SpinDynamics.jl is distributed under the terms of the repository's [`LICENSE`](LICENSE) file.
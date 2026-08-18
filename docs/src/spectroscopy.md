# Spectroscopy

SpinDynamics.jl provides static and dynamical spin structure factors through a high-level interface.

## Dynamical structure factor

A typical zero-temperature calculation starts from the ground state:

```julia
using SpinDynamics

L = 20

model = XXZChain(
    L;
    Jxy = 1.0,
    Jz = 1.0,
    nup = L ÷ 2,
)

E0, ψ0 = groundstate(model)

q = momenta(model)
ω = collect(range(0.0, 5.0; length = 100))
```

The same public interface can then use either Lanczos or KPM.

### Lanczos

```julia
S_lanczos = dynamical_structure_factor(
    model,
    ψ0,
    q,
    ω;
    method = :lanczos,
    lanc_m = 100,
    eta = 0.05,
)
```

### Kernel Polynomial Method

```julia
S_kpm = dynamical_structure_factor(
    model,
    ψ0,
    q,
    ω;
    method = :kpm,
    kpm_m = 80,
    kernel = :jackson,
)
```

Both approaches describe the same spectral structure but use different numerical representations and broadening mechanisms.

```@raw html
<div style="display: flex; gap: 1rem; align-items: flex-start;">
  <div style="width: 50%; text-align: center;">
    <strong>Lanczos</strong><br>
    <img src="../assets/lanczos_xxz_spectra_L20_Sz0.png" style="width:100%;">
  </div>
  <div style="width: 50%; text-align: center;">
    <strong>KPM</strong><br>
    <img src="../assets/kpm_xxz_spectra_L20_Sz0.png" style="width:100%;">
  </div>
</div>
```

## Static structure factor

The static structure factor is available through [`structure_factor`](@ref):

```julia
Sq = structure_factor(model, ψ0)
```

## Momentum grid

For translationally invariant chains, the discrete lattice momenta can be obtained with:

```julia
q = momenta(model)
```

The returned values correspond to the lattice momenta compatible with the system size.
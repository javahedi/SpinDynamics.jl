module FriendlyAPI

using ..SpinModel
using ..Hamiltonian
using ..Lanczos
using ..TimeEvolution


export groundstate
export groundstate, time_evolve

"""
    groundstate(model; method=:lanczos, kwargs...)

Compute the ground-state energy and wavefunction.

Currently supported methods:
- `:lanczos`
"""
function groundstate(model::SpinModel.Model; method::Symbol=:lanczos, kwargs...)
    if method === :lanczos
        return Lanczos.lanczos_groundstate(
            Hamiltonian.apply_H!,
            model;
            kwargs...
        )
    end

    throw(ArgumentError("unsupported ground-state method: $method"))
end





"""
    time_evolve(model, ψ0, t; method=:krylov, kwargs...)

Evolve a state by time `t`.

Currently supported methods:
- `:krylov`
- `:chebyshev`
"""
function time_evolve(
    model::SpinModel.Model,
    ψ0::AbstractVector,
    t::Real;
    method::Symbol=:krylov,
    Ebounds=nothing,
    kwargs...
)
    if method === :krylov
        return TimeEvolution.krylov_time_evolve(
            ψ0,
            Float64(t),
            Hamiltonian.apply_H!,
            model;
            kwargs...
        )

    elseif method === :chebyshev
        bounds = if isnothing(Ebounds)
            Lanczos.estimate_energy_bounds(
                Hamiltonian.apply_H!,
                model,
            )
        else
            Ebounds
        end

        return TimeEvolution.chebyshev_time_evolve(
            ψ0,
            Float64(t),
            Hamiltonian.apply_H!,
            model;
            Ebounds=bounds,
            kwargs...
        )
    end

    throw(ArgumentError("unsupported time-evolution method: $method"))
end

end
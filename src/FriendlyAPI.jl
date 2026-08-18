module FriendlyAPI

using ..SpinModel
using ..Hamiltonian
using ..Lanczos
using ..TimeEvolution

using ..Observables

export groundstate, time_evolve, structure_factor


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




"""
    structure_factor(model, ψ)

Compute the static spin structure factor `S(q)`.

Returns the same momentum-to-value mapping as the underlying
`structure_factor_Sq` implementation.
"""
function structure_factor(
    model::SpinModel.Model,
    ψ::AbstractVector,
)
    return Observables.structure_factor_Sq(ψ, model)
end

end
module FriendlyAPI

using ..SpinModel
using ..Hamiltonian
using ..Lanczos

export groundstate

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

end
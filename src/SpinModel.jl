module SpinModel

    export Model, build_model, nn_hopping, long_range_hopping, XXZChain, momenta
    using ..Basis

    mutable struct Model
        L::Int                      # number of spins
        nup::Union{Nothing,Int}     # nothing = full, Int = sector
        mode::Symbol                 # :full or :sector
        states::Vector{UInt64}
        idxmap::Dict{UInt64,Int}
        hopping_list::Vector{Tuple{Int,Int,Float64}}   # S⁺S⁻ or XY
        onsite_field::Vector{Float64}                  # local fields
        zz_list::Vector{Tuple{Int,Int,Float64}}       # SzSz couplings
    end

    """
        build_model(L; nup=nothing, hopping=[], h=zeros(L), zz=[])

    Create a spin model with given parameters. If `nup` is provided,
    the basis is restricted to that Sᶻ sector.
    """
    function build_model(L::Int; 
                        nup::Union{Nothing,Int}=nothing, 
                        hopping=[], onsite_field=zeros(L), zz=[])
        if isnothing(nup)
            states, idxmap = build_full_basis(L)
            mode = :full
        else
            states, idxmap = build_sector_basis(L, nup)
            mode = :sector
        end

        return Model(L, nup, mode, states, idxmap,
                    [(i,j,J) for (i,j,J) in hopping],
                    Vector{Float64}(onsite_field),
                    [(i,j,Jz) for (i,j,Jz) in zz])
    end

    function nn_hopping(L::Int, J::Float64)
        [(i, i+1, J) for i in 1:(L-1)]
    end

    function long_range_hopping(L::Int, J::Function)
        [(i, j, J(i,j)) for i in 1:L for j in i+1:L]
    end



    """
        XXZChain(L; Jxy=1.0, Jz=1.0, hz=0.0, nup=nothing, boundary=:open)

    Construct a nearest-neighbor spin-1/2 XXZ chain.

    # Arguments
    - `L`: Number of lattice sites.
    - `Jxy`: XY exchange coupling.
    - `Jz`: Ising coupling.
    - `hz`: Uniform longitudinal field.
    - `nup`: Number of up spins. If `nothing`, use the full Hilbert space.
    - `boundary`: `:open` or `:periodic`.
    """
    function XXZChain(
        L::Int;
        Jxy::Real=1.0,
        Jz::Real=1.0,
        hz::Real=0.0,
        nup::Union{Nothing,Int}=nothing,
        boundary::Symbol=:open,
    )
        hopping = [(i, i + 1, Float64(Jxy) / 2 )  for i in 1:(L - 1)]
        zz = [(i, i + 1, Float64(Jz)) for i in 1:(L - 1)]

        if boundary === :periodic
            if L > 2
                push!(hopping, (L, 1, Float64(Jxy) / 2) )
                push!(zz, (L, 1, Float64(Jz)))
            end
        elseif boundary !== :open
            throw(ArgumentError("boundary must be :open or :periodic"))
        end

        return build_model(
            L;
            nup=nup,
            hopping=hopping,
            onsite_field=fill(Float64(hz), L),
            zz=zz,
        )
    end

    """
        momenta(model)

    Return the lattice momenta `q = 2πn/L` for `n = 0, …, L-1`.
    """
    function momenta(model::Model)
        return 2π .* (0:(model.L - 1)) ./ model.L
    end



end # module

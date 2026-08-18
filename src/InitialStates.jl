module InitialStates

    using ..SpinModel
    export domain_wall_state, neel_state, polarized_state, polarized_state_with_flips

    # -------------------------------------------------
    # Domain wall state
    # -------------------------------------------------
    function domain_wall_state(model::SpinModel.Model)
        s = UInt64(0)

        nup = model.mode === :sector ?
            model.nup :
            Int(ceil(model.L / 2))

        for i in 0:(nup - 1)
            s |= UInt64(1) << i
        end

        ψ0 = zeros(Float64, length(model.states))

        idx = if model.mode === :full
            Int(s) + 1
        else
            get(model.idxmap, s, 0)
        end

        idx != 0 || throw(ArgumentError(
            "domain-wall state is not contained in the model basis"
        ))

        ψ0[idx] = 1.0
        return ψ0
    end


    # -------------------------------------------------
    # Néel state: ↑↓↑↓... (starting with ↑ at site 1)
    # -------------------------------------------------
   function neel_state(model::SpinModel.Model)
        s = UInt64(0)

        for i in 0:(model.L - 1)
            if isodd(i + 1)
                s |= UInt64(1) << i
            end
        end

        ψ0 = zeros(Float64, length(model.states))

        idx = if model.mode === :full
            Int(s) + 1
        else
            get(model.idxmap, s, 0)
        end

        idx != 0 || throw(ArgumentError(
            "Néel state is not contained in the model basis"
        ))

        ψ0[idx] = 1.0
        return ψ0
    end



    # -------------------------------------------------
    # Polarized state: all ↑ or all ↓
    # -------------------------------------------------
    function polarized_state(model::SpinModel.Model; up::Bool=true)
        L = model.L

        s = up ? (UInt64(1) << L) - UInt64(1) : UInt64(0)

        ψ0 = zeros(Float64, length(model.states))

        idx = if model.mode === :full
            Int(s) + 1
        else
            get(model.idxmap, s, 0)
        end

        idx != 0 || throw(ArgumentError(
            "requested polarized state is not contained in the model basis"
        ))

        ψ0[idx] = 1.0
        return ψ0
    end


    
    # -------------------------------------------------
    # Polarized with flipped sites
    # (start FM ↑, then flip given sites)
    # -------------------------------------------------
    function polarized_state_with_flips(model::SpinModel.Model, flips::Vector{Int})
        if model.mode == :sector
            s = (UInt64(1) << model.L) - 1  # all ↑
            for i in flips
                s ⊻= (UInt64(1) << (i-1))  # flip site i (1-based)
            end
            ψ0 = zeros(Float64, length(model.states))
            ψ0[model.idxmap[s]] = 1.0
            return ψ0
        else
            s = (UInt64(1) << model.L) - 1
            for i in flips
                s ⊻= (UInt64(1) << (i-1))
            end
            return s
        end
        
    end



end # module

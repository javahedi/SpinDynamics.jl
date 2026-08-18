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
    function polarized_state_with_flips(
            model::SpinModel.Model,
            flips::Vector{Int},
        )
            L = model.L

            for site in flips
                1 <= site <= L || throw(
                    ArgumentError("flip site $site is outside the model with L=$L")
                )
            end

            # Start from |↑↑...↑⟩
            s = (UInt64(1) << L) - UInt64(1)

            for site in flips
                s ⊻= UInt64(1) << (site - 1)
            end

            ψ0 = zeros(Float64, length(model.states))

            idx = if model.mode === :full
                Int(s) + 1
            else
                get(model.idxmap, s, 0)
            end

            idx != 0 || throw(ArgumentError(
                "requested flipped polarized state is not contained in the model basis"
            ))

            ψ0[idx] = 1.0
            return ψ0
        end



end # module

module InitialStates

    using ..SpinModel
    export domain_wall_state, neel_state, polarized_state, polarized_state_with_flips

    # -------------------------------------------------
    # Domain wall state
    # -------------------------------------------------
    function domain_wall_state(model::SpinModel.Model)
        if model.mode == :sector
            s = UInt64(0)
            nup = model.nup
            for i in 0:nup-1
                s |= UInt64(1) << i
            end
            ψ0 = zeros(Float64, length(model.states))
            ψ0[model.idxmap[s]] = 1.0
            return ψ0
        else
            # full Hilbert space
            s = UInt64(0)
            nup = Int(ceil(model.L / 2))
            for i in 0:(nup-1)
                s |= UInt64(1) << i
            end
            return s
        end
    end


    # -------------------------------------------------
    # Néel state: ↑↓↑↓... (starting with ↑ at site 1)
    # -------------------------------------------------
    function neel_state(model::SpinModel.Model)
        if model.mode == :sector
            s = UInt64(0)
            for i in 0:(model.L-1)
                if isodd(i+1)   # site index starting at 1
                    s |= UInt64(1) << i
                end
            end
            ψ0 = zeros(Float64, length(model.states))
            ψ0[model.idxmap[s]] = 1.0
            return ψ0
        else
            s = UInt64(0)
            for i in 0:(model.L-1)
                if isodd(i+1)
                    s |= UInt64(1) << i
                end
            end
            return s
        end
    end



    # -------------------------------------------------
    # Polarized state: all ↑ or all ↓
    # -------------------------------------------------
    function polarized_state(model::SpinModel.Model; up::Bool=true)
        L = model.L
        
        # Construct the fully polarized bitstring
        s_up   = (UInt64(1) << L) - UInt64(1)  # all spins up: 0b111…1
        s_down = UInt64(0)                       # all spins down: 0b000…0
        s = up ? s_up : s_down

        if model.mode == :sector
            # In sector mode, create a vector in the restricted basis
            ψ0 = zeros(Float64, length(model.states))
            
            # Check if the bitstring exists in the sector
            idx = get(model.idxmap, s, nothing)
            if idx === nothing
                error("The requested polarized state does not exist in this sector.")
            end
            
            ψ0[idx] = 1.0
            return ψ0
        else
            # Full Hilbert space: just return the bitstring
            return s
        end
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

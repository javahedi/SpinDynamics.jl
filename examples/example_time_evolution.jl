#!/usr/bin/env julia

using LinearAlgebra
using SpinDynamics
using Plots

function main()
    L = 15
    nup = L - 1

    model = XXZChain(
        L;
        Jxy = 1.0,
        Jz = 0.5,
        nup = nup,
    )

    middle_site = (L + 1) ÷ 2

    ψ0 = ComplexF64.(
        polarized_state_with_flips(model, [middle_site])
    )

    N = length(model.states)

    println("Hilbert-space dimension: ", N)
    println("Flipped site: ", middle_site)

    # Exact Hamiltonian
    H = zeros(ComplexF64, N, N)
    basis_vector = zeros(ComplexF64, N)
    column = zeros(ComplexF64, N)

    for j in 1:N
        fill!(basis_vector, 0)
        basis_vector[j] = 1

        apply_H!(column, basis_vector, model)
        H[:, j] .= column
    end

    # Time grid
    times = range(0.0, 5.0; length=150)
    dt = step(times)

    mags_exact = Matrix{Float64}(undef, L, length(times))
    mags_cheb = similar(mags_exact)
    mags_krylov = similar(mags_exact)

    fidelity_cheb = Vector{Float64}(undef, length(times))
    fidelity_krylov = similar(fidelity_cheb)

    ψ_exact = copy(ψ0)
    ψ_cheb = copy(ψ0)
    ψ_krylov = copy(ψ0)

    mags_exact[:, 1] = magnetization_per_site(ψ_exact, model)
    mags_cheb[:, 1] = mags_exact[:, 1]
    mags_krylov[:, 1] = mags_exact[:, 1]

    fidelity_cheb[1] = 1.0
    fidelity_krylov[1] = 1.0

    U = exp(-im * dt * H)

    @time for n in 1:(length(times) - 1)
        # Exact
        ψ_exact = U * ψ_exact
        ψ_exact ./= norm(ψ_exact)

        # Chebyshev
        ψ_cheb = time_evolve(
            model,
            ψ_cheb,
            dt;
            method = :chebyshev,
            cheb_n = 20,
        )

        # Krylov
        ψ_krylov = time_evolve(
            model,
            ψ_krylov,
            dt;
            method = :krylov,
            kry_m = 15,
        )

        mags_exact[:, n + 1] =
            magnetization_per_site(ψ_exact, model)

        mags_cheb[:, n + 1] =
            magnetization_per_site(ψ_cheb, model)

        mags_krylov[:, n + 1] =
            magnetization_per_site(ψ_krylov, model)

        fidelity_cheb[n + 1] =
            abs2(dot(ψ_exact, ψ_cheb))

        fidelity_krylov[n + 1] =
            abs2(dot(ψ_exact, ψ_krylov))
    end

    # Plotting
    plt1 = heatmap(
        1:L,
        times,
        -1.0.*mags_exact';
        xlabel = "Site",
        ylabel = "Time",
        title = "Exact",
        colorbar_title = "⟨Sᶻ⟩",
    )

    plt2 = heatmap(
        1:L,
        times,
        -1.0.*mags_cheb';
        xlabel = "Site",
        ylabel = "Time",
        title = "Chebyshev",
        colorbar_title = "⟨Sᶻ⟩",
    )

    plt3 = heatmap(
        1:L,
        times,
        -1.0.*mags_krylov';
        xlabel = "Site",
        ylabel = "Time",
        title = "Krylov",
        colorbar_title = "⟨Sᶻ⟩",
    )

    plt4 = plot(
        times,
        mags_exact[middle_site, :];
        label = "Exact",
        xlabel = "Time",
        ylabel = "⟨Sᶻ⟩",
        title = "Middle-site magnetization",
        linewidth = 2,
    )

    plot!(
        plt4,
        times,
        mags_cheb[middle_site, :];
        label = "Chebyshev",
        linestyle = :dash,
        linewidth = 2,
    )

    plot!(
        plt4,
        times,
        mags_krylov[middle_site, :];
        label = "Krylov",
        linestyle = :dot,
        linewidth = 2,
    )

    # plt4 = plot(
    #     times[2:end],
    #     1 .- fidelity_cheb[2:end];
    #     label = "Chebyshev",
    #     xlabel = "Time",
    #     ylabel = "1 - Fidelity",
    #     title = "Error vs exact",
    #     yscale = :log10,
    # )

    # plot!(
    #     plt4,
    #     times[2:end],
    #     1 .- fidelity_krylov[2:end];
    #     label = "Krylov",
    # )

    plt = plot(
        plt1,
        plt2,
        plt3,
        plt4;
        layout = (2, 2),
        size = (1200, 800),
    )

    outfile = joinpath(
        @__DIR__,
        "time_evolution_L$(L)_nup$(nup).png",
    )

    savefig(plt, outfile)

    println("Saved figure to: ", outfile)
    println("Minimum Chebyshev fidelity: ", minimum(fidelity_cheb))
    println("Minimum Krylov fidelity:    ", minimum(fidelity_krylov))
end

main()

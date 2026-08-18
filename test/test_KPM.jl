using Test
using SpinDynamics

@testset "KPM rescaling contains estimated spectrum" begin
    model = XXZChain(6; Jxy=1.0, Jz=1.0, nup=3)

    Emin, Emax = estimate_energy_bounds(
        apply_H!,
        model;
        lanc_m=20,
    )

    a, b = SpinDynamics.KPM_Sqw.get_rescaling_params(
        apply_H!,
        model;
        lanc_m=20,
    )

    xmin = (Emin - b) / a
    xmax = (Emax - b) / a

    @test xmin > -1.0
    @test xmax < 1.0
end


@testset "KPM rescaling keeps spectral bounds inside [-1, 1]" begin
    Emin = -3.0
    Emax = 5.0

    a, b = SpinDynamics.KPM_Sqw._rescaling_from_bounds(Emin, Emax)

    xmin = (Emin - b) / a
    xmax = (Emax - b) / a

    @test xmin ≈ -0.99 atol=1e-12
    @test xmax ≈  0.99 atol=1e-12

    @test -1.0 < xmin < 0.0
    @test  0.0 < xmax < 1.0
end


@testset "KPM spectral weight is on excitation-energy scale" begin
    model = XXZChain(6; Jxy=1.0, Jz=1.0, nup=3)
    E0, ψ0 = groundstate(model; lanc_m=20)

    q = [Float64(π)]
    ω = collect(range(0.0, 5.0; length=300))

    S = dynamical_structure_factor(
        model,
        ψ0,
        q,
        ω;
        method=:kpm,
        kpm_m=100,
    )

    @test all(isfinite, S)
    @test all(S .>= 0.0)

    # No artificial edge plateau at the top of the requested ω window.
    @test maximum(S[:, end-10:end]) < maximum(S)
end

@testset "KPM dynamical structure-factor sum rule" begin
    model = XXZChain(6; Jxy=1.0, Jz=1.0, nup=3)
    _, ψ0 = groundstate(model; lanc_m=20)

    q = Float64(π)
    ω = collect(0.0:0.01:5.0)
    dω = ω[2] - ω[1]

    phi = Sz_q_vector(model, ψ0, q)
    exact_weight = norm(phi)^2

    S = dynamical_structure_factor(
        model,
        ψ0,
        [q],
        ω;
        method=:kpm,
        kpm_m=120,
        kernel=:jackson,
    )

    integrated_weight = sum(S[1, :]) * dω

    @test integrated_weight ≈ exact_weight rtol=5e-3
end
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
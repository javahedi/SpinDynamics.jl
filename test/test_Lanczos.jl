using Test
using LinearAlgebra
using SpinDynamics

@testset "Lanczos tridiagonalization with complex vector" begin
    model = XXZChain(2; Jxy=1.0, Jz=1.0, nup=1)

    v = ComplexF64[1.0, im]
    v ./= norm(v)

    Hv = similar(v)
    apply_H!(Hv, v, model)

    α_expected = real(dot(v, Hv))

    α, β, normv = lanczos_tridiag(
        apply_H!,
        model,
        v;
        lanc_m=2,
    )

    @test α[1] ≈ α_expected atol=1e-12
    @test normv ≈ 1.0 atol=1e-12
end
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


@testset "Lanczos ground state against exact diagonalization" begin
    model = XXZChain(6; Jxy=1.0, Jz=1.0, nup=3)

    N = length(model.states)

    H = zeros(Float64, N, N)
    e = zeros(Float64, N)
    out = zeros(Float64, N)

    for j in 1:N
        fill!(e, 0.0)
        e[j] = 1.0
        apply_H!(out, e, model)
        H[:, j] .= out
    end

    E_exact = minimum(eigvals(H))

    E0, ψ0 = groundstate(model; lanc_m=N)

    @test E0 ≈ E_exact atol=1e-12
    @test norm(ψ0) ≈ 1.0 atol=1e-12
    @test norm(H * ψ0 - E0 * ψ0) < 1e-10
end
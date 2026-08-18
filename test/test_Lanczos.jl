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



@testset "Lanczos dimension is capped by Hilbert space" begin
    model = XXZChain(4; nup=2)

    N = length(model.states)

    E0, ψ0 = groundstate(model; lanc_m=100)

    @test length(ψ0) == N
    @test norm(ψ0) ≈ 1.0 atol=1e-12

    Hψ = similar(ψ0)
    apply_H!(Hψ, ψ0, model)

    @test norm(Hψ - E0 * ψ0) < 1e-10
end


@testset "Lanczos extremal dimension is capped" begin
    model = XXZChain(4; Jxy=1.0, Jz=1.0, nup=2)

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

    exact = eigvals(Hermitian(H))

    Emin, Emax = lanczos_extremal(
        apply_H!,
        model;
        lanc_m=100,
    )

    @test Emin ≈ first(exact) atol=1e-12
    @test Emax ≈ last(exact) atol=1e-12
end


@testset "Lanczos tridiagonal dimension is capped" begin
    model = XXZChain(4; nup=2)
    N = length(model.states)

    v = randn(ComplexF64, N)
    v ./= norm(v)

    α, β, _ = lanczos_tridiag(
        apply_H!,
        model,
        v;
        lanc_m=100,
    )

    @test length(α) <= N
    @test length(β) == length(α) - 1
end
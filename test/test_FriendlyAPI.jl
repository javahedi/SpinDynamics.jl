using Test
using LinearAlgebra
using SpinDynamics

@testset "Friendly API: XXZChain" begin
    model = XXZChain(2; Jxy=1.0, Jz=1.0, nup=1)

    @test model.L == 2
    @test model.nup == 1
    @test model.mode == :sector
    @test length(model.states) == 2

    H = zeros(Float64, 2, 2)
    e = zeros(Float64, 2)
    out = zeros(Float64, 2)

    for j in 1:2
        fill!(e, 0.0)
        e[j] = 1.0
        apply_H!(out, e, model)
        H[:, j] .= out
    end

    @test H ≈ [-0.25 0.5;
                0.5 -0.25]

    @test eigvals(H) ≈ [-0.75, 0.25]
end

@testset "Friendly API: momenta" begin
    model = XXZChain(6; nup=3)

    q = momenta(model)

    @test length(q) == 6
    @test q ≈ 2π .* (0:5) ./ 6
end


@testset "Friendly API: groundstate" begin
    model = XXZChain(2; Jxy=1.0, Jz=1.0, nup=1)

    E0, ψ0 = groundstate(model; lanc_m=2)

    @test E0 ≈ -0.75 atol=1e-12
    @test norm(ψ0) ≈ 1.0 atol=1e-12

    Hψ = similar(ψ0)
    apply_H!(Hψ, ψ0, model)

    @test norm(Hψ - E0 * ψ0) < 1e-10
    @test_throws ArgumentError groundstate(model; method=:unknown)
end
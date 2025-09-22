using Test
using SpinDynamics
using LinearAlgebra

@testset "Hamiltonian Tests" begin
    L = 4
    nup = 2

    # Build models
    model_full = build_model(L)
    model_sector = build_model(L; nup=nup)

    # ------------------------
    # Bit helpers
    # ------------------------
    @test bit_at(UInt64(0x05), 0) == 1
    @test bit_at(UInt64(0x05), 1) == 0
    @test sz_value(UInt64(1)) == 0.5
    @test sz_value(UInt64(0)) == -0.5
    @test flip_bits(UInt64(0x05), 0, 1) == UInt64(0x06)

   

    # ------------------------
    # Spin operators
    # ------------------------
    ψ0_full = zeros(ComplexF64, 1<<L)
    ψ0_full[1] = 1.0 + 0im

    Sz1 = create_spin_operator(1, :z)
    ψ_sz1 = Sz1(ψ0_full, model_full)
    @test ψ_sz1[1] == -0.5 + 0im

    Splus1 = create_spin_operator(1, :plus)
    ψ_plus1 = Splus1(ψ0_full, model_full)
    @test ψ_plus1[2] == 1.0 + 0im

    Sx1 = create_spin_operator(1, :x)
    ψ_x1 = Sx1(ψ0_full, model_full)
    @test ψ_x1[2] == 0.5 + 0im

    Sy1 = create_spin_operator(1, :y)
    ψ_y1 = Sy1(ψ0_full, model_full)
    @test ψ_y1[2] == -0.5im

    # ------------------------
    # Full Hamiltonian
    # ------------------------
    hopping = [(1,2,1.0)]
    zz = [(1,2,0.5)]
    h = zeros(L)
    model_test = build_model(L; hopping=hopping, onsite_field=h, zz=zz)

    ψ_in = zeros(Float64, 1<<L)
    ψ_in[1] = 1.0
    ψ_out = similar(ψ_in)
    apply_H!(ψ_out, ψ_in, model_test)

    
end

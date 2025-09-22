
using Test
using SpinDynamics
using LinearAlgebra

# ----------------------------
# Test SpinModel
# ----------------------------
@testset "SpinModel Tests" begin

    # Test build_model for full Hilbert space
    L = 4
    model_full = build_model(L)
    @test model_full.L == L
    @test model_full.mode == :full
    @test isa(model_full.states, Vector{UInt64})
    @test isa(model_full.idxmap, Dict)

    # Check that the number of states is 2^L
    @test length(model_full.states) == 2^L

    # Test build_model for sector
    nup = 2
    model_sector = build_model(L; nup=nup)
    @test model_sector.mode == :sector
    @test all(count_ones(s) == nup for s in model_sector.states)

    # Test nearest-neighbor hopping
    J = 1.0
    nn_list = nn_hopping(L, J)
    @test length(nn_list) == L-1
    @test all(length(t) == 3 && t[3] == J for t in nn_list)

    # Test long-range hopping
    Jfun(i,j) = 1.0 / (abs(i-j)^3)
    lr_list = long_range_hopping(L, Jfun)
    @test length(lr_list) == L*(L-1) ÷ 2
    @test all(t[3] ≈ Jfun(t[1], t[2]) for t in lr_list)

    # Test build_model with hopping and onsite_field
    h = rand(L)
    hopping = nn_list
    zz = [(1,2,0.5)]
    model_test = build_model(L; nup=nothing, hopping=hopping, onsite_field=h, zz=zz)
    @test model_test.hopping_list == hopping
    @test model_test.onsite_field == h
    @test model_test.zz_list == zz

end

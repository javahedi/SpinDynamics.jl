using Test
using Aqua
using SpinDynamics

@testset "Aqua" begin
    Aqua.test_all(SpinDynamics)
end
@testset "test_examples" begin


script_path = "../examples/filtering.jl"

@test test_script_crash(script_path)

A = 1
B = 2
C = 3

@test A == A

@test A+B == C

@test C-A == B

end

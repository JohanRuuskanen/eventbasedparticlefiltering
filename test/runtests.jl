using Test
using Random
using EventBasedParticleFiltering

include("framework.jl")

my_tests = ["test_examples",
            "test_plotting"]


run_tests(my_tests)

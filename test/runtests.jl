using Test
using Random
using LinearAlgebra
using EventBasedParticleFiltering

include("framework.jl")

my_tests = ["test_examples",
            "test_plotting",
            "test_misc"]


run_tests(my_tests)

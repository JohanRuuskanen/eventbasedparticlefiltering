using Test
using Random
using Statistics
using LinearAlgebra
using EventBasedParticleFiltering

const EP = EventBasedParticleFiltering

include("framework.jl")

my_tests = ["test_examples",
            "test_plotting",
            "test_misc",
            "test_resampling"]

run_tests(my_tests)

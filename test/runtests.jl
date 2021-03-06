using Test
using Random
using Statistics
using Distributions
using LinearAlgebra
using EventBasedParticleFiltering

const EP = EventBasedParticleFiltering

include("framework.jl")

my_tests = ["test_examples",
            "test_plotting",
            "test_misc",
            "test_propagation",
            "test_weighting",
            "test_resampling",
            "test_events",
            "test_filtering"]

run_tests(my_tests)

using Test
using Random
using Calculus
using StatsBase
using Statistics
using Distributions
using LinearAlgebra
using EventBasedParticleFiltering

const EP = EventBasedParticleFiltering

include("framework.jl")

my_tests = ["test_resampling",
            "test_misc",
            "test_proposals",
            "test_events",
            "test_weighting",
            "test_filtering",
            "test_predcomp"]

run_tests(my_tests)

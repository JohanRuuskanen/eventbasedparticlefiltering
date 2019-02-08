module EventBasedParticleFiltering

    using JLD
    using PyPlot
    using Random
    using StatsBase
    using Distributions
    using LinearAlgebra

    include("linear_systems/filters.jl")
    include("linear_systems/filters_eventbased.jl")

    include("funcs/misc.jl")
    include("funcs/plotting.jl")

end

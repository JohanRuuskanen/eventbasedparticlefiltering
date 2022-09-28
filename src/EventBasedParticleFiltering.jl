module EventBasedParticleFiltering

    using PyPlot
    using Random
    using Cubature
    using Calculus
    using StatsBase
    using Parameters
    using KernelDensity
    using Distributions
    using LinearAlgebra

    include("funcs/misc.jl")
    include("funcs/event_kernels.jl")
    include("funcs/resampling.jl")
    include("funcs/weighting.jl")
    include("funcs/filters.jl")
    include("funcs/filters_eventbased.jl")
    include("funcs/plotting_funcs.jl")
    include("funcs/proposals.jl")
    include("funcs/propagation.jl")

    export  compute_err_metrics,
            ebpf,
            ebpf_options,
            kernel_IBT,
            kernel_SOD,
            likelihood_analytic,
            likelihood_cubature,
            likelihood_MC,
            plot_measurement_data,
            plot_particle_trace,
            plot_particle_hist,
            pftype_bootstrap,
            pftype_auxiliary,
            sim_sys,
            system

end

module EventBasedParticleFiltering

    export  sys_params,
            lin_sys_params,
            output,
            pf_params,
            sim_sys,
            sim_lin_sys,
            fix_sym,
            plot_particle_trace,
            plot_data,
            plot_effective_sample_size,
            ebpf,
            eapf,
            ebse,
            bpf,
            apf,
            kalman_filter,
            systematic_resampling

    using JLD
    using PyPlot
    using Random
    using StatsBase
    using Distributions
    using LinearAlgebra

    include("funcs/filters.jl")
    include("funcs/filters_eventbased.jl")
    include("funcs/misc.jl")
    include("funcs/plotting.jl")
    include("funcs/resampling.jl")

end

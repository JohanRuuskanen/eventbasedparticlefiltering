module EventBasedParticleFiltering

    println("Got here")

    export  sys_params,
            #lin_sys_params,
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
            propagation_bootstrap,
            propagation_locallyOptimal_newEvent,
            propagation_locallyOptimal_noEvent,
            eventSampling

    using JLD
    using PyPlot
    using Random
    using StatsBase
    using Distributions
    using LinearAlgebra

    println("Got here too")

    include("funcs/filters.jl")
    include("funcs/filters_eventbased.jl")
    include("funcs/misc.jl")
    include("funcs/event_kernels.jl")
    include("funcs/plotting.jl")
    include("funcs/propagation.jl")
    include("funcs/resampling.jl")

end

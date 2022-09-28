
function ebpf(y::AbstractArray{T,2}, opt::ebpf_options;
                X0=Array{T,2}(undef,0,0)::AbstractArray{T,2}) where T <: AbstractFloat
    """
    Event-based particle filter
    """

    # Perform some assertions to catch errors early on

    @assert opt.N > 0
    @assert opt.sys.nx > 0
    @assert opt.sys.ny > 0
    @assert opt.sys.t_end > 0

    @assert opt.triggerat in ["events", "always", "never"]

    @assert all(opt.kernel.δ .>= 0)
    @assert length(opt.kernel.δ) == opt.sys.ny

    @assert try opt.sys.px(ones(opt.sys.nx), 1); true catch; false end
    @assert try opt.sys.py(ones(opt.sys.nx), 1); true catch; false end

    @assert typeof(opt.sys.px(ones(opt.sys.nx), 1)) <: Distribution
    @assert typeof(opt.sys.py(ones(opt.sys.nx), 1)) <: Distribution

    @assert size(mean(opt.sys.px(ones(opt.sys.nx), 1))) == (opt.sys.nx,)
    @assert size(mean(opt.sys.py(ones(opt.sys.nx), 1))) == (opt.sys.ny,)

    if typeof(opt.pftype) <: pftype_auxiliary
        @assert opt.pftype.D > 0

        @assert try opt.pftype.qv(ones(opt.sys.nx), 1, opt,
            V=Diagonal(ones(opt.sys.ny))); true catch; false end
        @assert try opt.pftype.q(ones(opt.sys.nx), 1, ones(opt.sys.ny), opt,
            V=Diagonal(ones(opt.sys.ny))); true catch; false end

        @assert typeof(opt.pftype.qv(ones(opt.sys.nx), 1, opt,
            V=Diagonal(ones(opt.sys.ny)))) <: Distribution
        @assert typeof(opt.pftype.q(ones(opt.sys.nx), 1, ones(opt.sys.ny), opt,
            V=Diagonal(ones(opt.sys.ny)))) <: Distribution

        @assert size(mean(opt.pftype.qv(ones(opt.sys.nx), 1, opt,
            V=Diagonal(ones(opt.sys.ny))))) == (opt.sys.ny,)
        @assert size(mean(opt.pftype.q(ones(opt.sys.nx), 1, ones(opt.sys.ny), opt,
            V=Diagonal(ones(opt.sys.ny))))) == (opt.sys.nx,)
    end

    # Allocations
    pfd = generate_pfd(opt, T=T)

    if !isempty(X0)
        pfd.X[:, :, 1] .= X0
    else
        pfd.X[:, :, 1] .= rand(Normal(0, 1), opt.N, opt.sys.nx)
    end

    pfd.W[:, 1] .= 1/opt.N .* ones(T, opt.N)
    pfd.S[:, 1] .= collect(1:opt.N)

    pfd.Xr[:, :, 1] .= pfd.X[:, :, 1]
    pfd.Xp[:, :, 1] .= pfd.X[:, :, 1]

    pfd.V[:, 1] .= zeros(T, opt.N)

    pfd.Γ[1] = 1

    # Filtering
    if opt.predictive_computation
        ebpf_predpost!(pfd, y, opt)
    else
        ebpf_ordinary!(pfd, y, opt)
    end

    return pfd
end

function ebpf_ordinary!(pfd::particle_data, y::AbstractArray{T,2},
    opt::ebpf_options) where T <: AbstractFloat

    for k = 2:opt.sys.t_end

        if opt.print_progress
            print("Running $(k) / $(opt.sys.t_end) \r")
            flush(stdout)
        end

        generate_Hk!(pfd, y, k, opt)

        eventtrigger!(pfd, y, k, opt)

        create_qv_list!(pfd, y, k, opt)

        resample!(pfd, k, opt)

        create_q_list!(pfd, y, k, opt)

        propagate!(pfd, k, opt)

        calculate_weights!(pfd, y, k, opt)

        if pfd.Γ[k] == 1 && opt.abort_at_trig
            break
        end

    end
end

function ebpf_predpost!(pfd::particle_data, y::AbstractArray{T,2},
    opt::ebpf_options) where T <: AbstractFloat

    @assert haskey(opt.extra_params, "a") "Predictive computation needs quantile parameter a"
    @assert 0.0 < opt.extra_params["a"] < 1.0 "Quantile parameter a needs to be in (0, 1)"

    function precompute!(k::Integer, pfd::particle_data, y::AbstractArray{T,2},
        opt::ebpf_options)

        p_tmp = 1
        p_first = 0
        p = 0

        y_dummy = zeros(typeof(first(pfd.X)), 0, 0)
        n = k

        nh = opt.debug_save ? n : 1
        nh_prev = opt.debug_save ? n-1 : 1

        while p < opt.extra_params["a"] && n < opt.sys.t_end
            n += 1

            pfd.Γ[n] = 0

            # need y to create SOD boundaries first step after a trigger
            generate_Hk!(pfd, y, n, opt)

            create_qv_list!(pfd, y_dummy, n, opt)

            resample!(pfd, n, opt)

            create_q_list!(pfd, y_dummy, n, opt)

            propagate!(pfd, n, opt)

            calculate_weights!(pfd, y_dummy, n, opt)

            p_first = pfd.p_trig[n] * p_tmp
            p_tmp = p_tmp * (1 - pfd.p_trig[n])
            p += p_first

        end

        return n + 1
    end

    pfd.extra["triggerwhen"] = Array{Int64, 2}(undef, 0, 2)
    n_hat = Array{Int64, 1}(undef, 0)
    append!(n_hat, precompute!(1, pfd, y, opt))

    for k = 2:opt.sys.t_end
        #println("$k, $k_stop")
        if opt.print_progress
            print("Running $(k) / $(opt.sys.t_end) \r")
            flush(stdout)
        end

        kh = opt.debug_save ? n : 1
        kh_prev = opt.debug_save ? n-1 : 1

        eventtrigger!(pfd, y, k, opt)

        if k == n_hat[end]
            pfd.Γ[k] = 1
        end

        # Compute new proposal posterior once we trigger
        if pfd.Γ[k] == 1
            pfd.extra["triggerwhen"] = vcat(pfd.extra["triggerwhen"], [n_hat[end] k])

            create_qv_list!(pfd, y, k, opt)

            resample!(pfd, k, opt)

            create_q_list!(pfd, y, k, opt)

            propagate!(pfd, k, opt)

            calculate_weights!(pfd, y, k, opt)

            append!(n_hat, precompute!(k, pfd, y, opt))
        end

    end

    return pfd
end

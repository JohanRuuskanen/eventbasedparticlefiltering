
function calculate_weights!(pfd::particle_data, y::AbstractArray{T,2},
    k::Integer, opt::ebpf_options) where T <: AbstractFloat

    function calculate_py(X)
        py = zeros(opt.N)
        if pfd.Γ[k]
            for i = 1:opt.N
                py[i] = pdf(opt.sys.py(X[i, :], k), y[:, k])
            end
        else
            # Analytic expression if ny = 1, otherwise needs to be solved numerically
            if opt.sys.ny == 1 && typeof(opt.likelihood) <: likelihood_analytic
                for i = 1:opt.N
                    p_multi = opt.sys.py(X[i, :], k)
                    p = Normal(mean(p_multi)[1], sqrt(var(p_multi)[1]))
                    py[i] = cdf(p, pfd.H[1, 2, k]) - cdf(p, pfd.H[1, 1, k])
                end
            else
                if typeof(opt.likelihood) <: likelihood_analytic
                    error("Analytic integration not defined for ny > 1")
                elseif typeof(opt.likelihood) <: likelihood_MC
                    for i = 1:opt.N
                        yh = rand(opt.sys.py(X[i, :], k), opt.likelihood.M)
                        py[i] = sum(all(pfd.H[1, 1, k] .<= yh .<= pfd.H[1, 2, k], dims=1)) / opt.likelihood.M
                    end
                elseif typeof(opt.likelihood) <: likelihood_cubature
                    for i = 1:opt.N
                        f(x) = pdf(opt.sys.py(X[i, :], k), x)
                        py[i] = hcubature(f, pfd.H[:, 1, k], pfd.H[:, 2, k], reltol=opt.numInt.reltol)[1]
                    end
                end
            end

            # Sometimes the integration fails
            for i = 1:opt.N
                # Since the numerical integration can be shaky
                if py[i] < 0
                    #println("Warning, p_trig = $(py[i]). Likely cause is the numerical integration. Setting p_trig to 0")
                    py[i] = 0
                elseif py[i] > 1
                    #println("Warning, p_trig = $(py[i]). Likely cause is the numerical integration. Setting p_trig to 1")
                    py[i] = 1
                end
            end
        end
        return py
    end

    kh = opt.debug_save ? k : 1
    kh_prev = opt.debug_save ? k-1 : 1

    # Simultaneuos weight/triggering probability calculation
    if typeof(opt.pftype) <: pftype_bootstrap
        py = calculate_py(view(pfd.X, :, :, k))
        pfd.W[:, k] .= log.(py)

        if pfd.Γ[k] == 0
            pfd.p_trig[k] = mean(1 .- py)
        end
    elseif typeof(opt.pftype) <: pftype_auxiliary
        W_tmp = zeros(opt.N)
        for i = 1:opt.N
            W_tmp[i] =  log(pdf(opt.sys.px(pfd.Xr[i, :, kh_prev], k-1), pfd.X[i, :, k])) -
                        log(pdf(pfd.q_list[i, kh], pfd.X[i, :, k])) -
                        log(sum(pfd.qv_list[:, i, kh_prev]))

            # can occur if the approximate predictive likelihood is ill-conditioned,
            # then it should be set to some large value
            if W_tmp[i] == Inf
              W_tmp[i] = 10000
            end
        end

        pfd.W[:, k] .= W_tmp .+ log.(calculate_py(view(pfd.X, :, :, k)))

        if pfd.Γ[k] == 0
            if !(typeof(opt.kernel) <: kernel_IBT)
                estimate_py_pred!(pfd, k, opt)
            end
            pfd.p_trig[k] = mean(1 .- calculate_py(view(pfd.Xp, :, :, kh)))
        end
    else
        error("This error message should not be able to trigger")
    end

    normalize_weights!(view(pfd.W, :, k), pfd, k, opt)

end

function normalize_weights!(W::AbstractArray{T, 1}, pfd::particle_data, k::Integer,
    opt::ebpf_options) where T <: AbstractFloat

    w_max = maximum(W)
    if w_max > -Inf
        W_norm = sum(exp.(W - w_max*ones(opt.N)))
        W .= exp.(W - w_max*ones(opt.N)) ./ W_norm
    else
        if opt.print_progress
            println("Weights at k=$(k) all 0 for EBPF! Resetting to uniform")
        end
        W .= 1/opt.N*ones(opt.N)
        pfd.fail[k] = true
    end

end

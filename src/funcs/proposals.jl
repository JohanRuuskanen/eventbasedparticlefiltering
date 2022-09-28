function genprop_linear_gaussian_noise()
    function q(x::AbstractArray{T,1}, k::Integer, y::AbstractArray{T,1},
         opt::ebpf_options; V=0.0) where T <: AbstractFloat

        Ck = zeros(opt.sys.ny, opt.sys.nx)
        for i = 1:opt.sys.ny
            Ck[i, :] .= Calculus.gradient(x -> mean(opt.sys.py(x[1:end-1], x[end]))[i],
                                            vcat(mean(opt.sys.px(x, k)), k+1))[1:end-1]
        end

        μ1 = mean(opt.sys.px(x, k))
        μ2 = mean(opt.sys.py(mean(opt.sys.px(x, k)), k+1))
        Q = cov(opt.sys.px(x, k))
        R = cov(opt.sys.py(mean(opt.sys.px(x, k)), k+1))

        Σ11 = Q
        Σ12 = Q*Ck'
        Σ21 = Ck*Q
        Σ22 = Ck*Q*Ck' + R .+ V

        return MvNormal(μ1 + Σ12 * (Σ22 \ (y - μ2)), fix_sym(Σ11 - Σ12 * (Σ22 \ Σ21)))
    end

    function qv(x::AbstractArray{T,1}, k::Integer, opt::ebpf_options;
        V=[]) where T <: AbstractFloat

        if isempty(V)
            V = zeros(opt.sys.ny, opt.sys.ny)
        end

        μ2 = mean(opt.sys.py(mean(opt.sys.px(x, k)), k+1))
        Q = cov(opt.sys.px(x, k))
        R = cov(opt.sys.py(mean(opt.sys.px(x, k)), k+1))

        Ck = zeros(opt.sys.ny, opt.sys.nx)
        for i = 1:opt.sys.ny
            Ck[i, :] = Calculus.gradient(x -> mean(opt.sys.py(x[1:end-1], x[end]))[i],
                                            vcat(mean(opt.sys.px(x, k)), k+1))[1:end-1]
        end

        Σ22 = Ck*Q*Ck' + R + V

        return MvNormal(μ2, Σ22)
    end

    return q, qv
end

function genprop_EMM_for_nonlinearclassic()


    function q(x, k, y, opt; V=0.0)

        if isempty(V)
            V = zeros(opt.sys.ny, opt.sys.ny)
        end

        Q = cov(opt.sys.px(x, k))
        R = cov(opt.sys.py(mean(opt.sys.px(x, k)), k+1))

        μ1 = mean(opt.sys.px(x, k))
        μ2 = μ1.^2/20 .+ first(Q)/20
        Σ11 = Q
        Σ12 = μ1*Q / 10
        Σ21 = Σ12
        Σ22 = μ1.^2 * Q / 100 + Q.^2/200 + R .+ V

        return MvNormal(μ1 + Σ12 * (Σ22 \ (y - μ2)), fix_sym(Σ11 - Σ12 * (Σ22 \ Σ21)))
    end

    function qv(x, k, opt; V=0.0)

        Q = cov(opt.sys.px(x, k))
        R = cov(opt.sys.py(mean(opt.sys.px(x, k)), k+1))

        μ1 = mean(opt.sys.px(x, k))
        μ2 = μ1.^2/20 .+ first(Q)/20
        Σ22 = μ1.^2 * Q / 100 + Q.^2/200 + R .+ V

        return MvNormal(μ2, Σ22)
    end

    return q, qv
end


function create_qv_list!(pfd::particle_data, y::AbstractArray{T,2}, k::Integer,
    opt::ebpf_options) where T <: AbstractFloat

    if typeof(opt.pftype) <: pftype_bootstrap
        # Do nothing
    elseif typeof(opt.pftype) <: pftype_auxiliary

        kh = opt.debug_save ? k : 1
        kh_prev = opt.debug_save ? k-1 : 1

        if pfd.Γ[k]
            for i = 1:opt.N
                pfd.qv_list[:, i, kh_prev] .= pdf(opt.pftype.qv(pfd.X[i, :, k-1], k-1, opt), y[:, k])
                pfd.qv_list[:, i, kh_prev] ./= opt.pftype.D
            end
        else
            μh = mean.(pfd.Hh[:, kh])
            Vh = cov.(pfd.Hh[:, kh])
            for i = 1:opt.N
                for j = 1:opt.pftype.D
                    pfd.qv_list[j, i, kh_prev] = pdf(opt.pftype.qv(pfd.X[i, :, k-1], k-1, opt, V=Vh[j]), μh[j])
                end
                pfd.qv_list[:, i, kh_prev] ./= opt.pftype.D
            end
        end
    else
        error("This error should not happen")
    end

end

function create_q_list!(pfd::particle_data, y::AbstractArray{T,2}, k::Integer,
    opt::ebpf_options) where T <: AbstractFloat

    kh = opt.debug_save ? k : 1
    kh_prev = opt.debug_save ? k-1 : 1

    if typeof(opt.pftype) <: pftype_bootstrap
        if typeof(opt.kernel) <: kernel_IBT
            # do nothing
        else
            for i = 1:opt.N
                pfd.q_list[i, kh] = opt.sys.px(pfd.Xr[i, :, kh_prev], k-1)
            end
        end
    elseif typeof(opt.pftype) <: pftype_auxiliary
        if pfd.Γ[k]
            for i = 1:opt.N
                pfd.q_list[i, kh] = opt.pftype.q(pfd.Xr[i, :, kh_prev], k-1, y[:, k], opt)
            end
        else
            μh = mean.(pfd.Hh[:, kh])
            Vh = cov.(pfd.Hh[:, kh])
            for i = 1:opt.N
                pfd.q_list[i, kh] = MixtureModel(map((μ, V) ->
                    opt.pftype.q(pfd.Xr[i, :, kh_prev], k-1, μ, opt, V=V), μh, Vh),
                    pfd.qv_list[:, i, kh_prev] ./ sum(pfd.qv_list[:, i, kh_prev]))
            end
        end
    else
        error("This error should not happen")
    end

end

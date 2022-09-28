
function generate_Hk!(pfd::particle_data, y::AbstractArray{T,2}, k::Integer,
    opt::ebpf_options) where T <: AbstractFloat

    if typeof(opt.kernel) <: kernel_SOD

        if pfd.Γ[k-1] || k == 2
            pfd.H[:, :, k] = hcat(y[:, k-1] .- opt.kernel.δ, y[:, k-1] .+ opt.kernel.δ)
        else
            pfd.H[:, :, k] .= pfd.H[:, :, k-1]
        end

    elseif typeof(opt.kernel) <: kernel_IBT

        Y = estimate_py_pred!(pfd, k, opt)

        yh = 1/opt.N * sum(mean.(Y))
        if typeof(yh) <: AbstractFloat
            yh = [yh]
        end

        pfd.H[:, :, k] .= hcat(yh .- opt.kernel.δ, yh .+ opt.kernel.δ)

    else
        error("This error should not be able to trigger")
    end

    # If auxiliary filter, generate discretization of Hk
    # Only works for 1D measurements so far
    if typeof(opt.pftype) <: pftype_auxiliary
        if opt.pftype.D > 1
            μh = collect(range(pfd.H[1, 1, k], stop=pfd.H[1, 2, k], length=opt.pftype.D))
        else
            μh = (pfd.H[1, 2, k] - pfd.H[1, 1, k]) / 2
        end

        kh = opt.debug_save ? k : 1

        Vh = (pfd.H[1, 2, k] - pfd.H[1, 1, k])/opt.pftype.D * 0.5
        for j = 1:opt.pftype.D
            pfd.Hh[j, kh] =  MvNormal([μh[j]], Vh)
        end
    end

end

function eventtrigger!(pfd::particle_data, y::Array{T,2}, k::Integer,
        opt::ebpf_options) where T <: AbstractFloat

    if opt.triggerat == "events"
        if !all(pfd.H[:, 1, k] .< y[:, k] .< pfd.H[:, 2, k])
             pfd.Γ[k] = 1
        end
    elseif opt.triggerat == "always"
        pfd.Γ[k] = 1
    elseif opt.triggerat == "never"
        pfd.Γ[k] = 0
    else
        error("No support for this type of trigger meta-option")
    end

end

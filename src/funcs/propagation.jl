function propagate!(pfd::particle_data, k::Integer, opt::ebpf_options)
    kh = opt.debug_save ? k : 1
    kh_prev = opt.debug_save ? k-1 : 1

    if typeof(opt.pftype) <: pftype_bootstrap && typeof(opt.kernel) <: kernel_IBT
        pfd.X[:, :, k] .= pfd.Xp[:, :, kh]
    else
        for i = 1:opt.N
            pfd.X[i, :, k] .= rand(pfd.q_list[i, kh])
        end
    end
end

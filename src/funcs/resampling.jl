function resample!(pfd::particle_data, k::Integer, opt::ebpf_options)

    kh = opt.debug_save ? k : 1
    kh_prev = opt.debug_save ? k-1 : 1

    if typeof(opt.pftype) <: pftype_bootstrap && typeof(opt.kernel) <: kernel_IBT

        if pfd.S[1, k] == 0
            pfd.S[:, k] .= resampling_systematic(pfd.W[:, k-1])
        end

        pfd.Xr[:, :, kh_prev] .= pfd.X[pfd.S[:, k], :, k-1]
    else
        pfd.V[:, kh_prev] .= pfd.W[:, k-1] .* sum(pfd.qv_list[:, :, kh_prev], dims=1)[:]
        idx = resampling_systematic(pfd.V[:, kh_prev])

        pfd.Xr[:, :, kh_prev] = pfd.X[idx, :, k-1]
        pfd.qv_list[:, :, kh_prev] = pfd.qv_list[:, idx, kh_prev]

        pfd.S[:, k] .= idx
    end

end


function resampling_multinomial(W::Array{T,1}) where T <: AbstractFloat
    N = size(W, 1)
    idx = rand(Categorical(W), N)
    return idx
end

# Might work poorly for low numerical precision, e.g. Float16
function resampling_systematic(W::Array{T,1}) where T <: AbstractFloat
    N = size(W, 1)
    idx = collect(1:N)
    wc = cumsum(W)

    wc ./= wc[end]

    u = ((collect(0:(N-1)) .+ rand()) / N)
    c = 1
    for i = 1:N
        while wc[c] < u[i]
            c = c + 1
        end
        idx[i] = c
    end
    return idx
end

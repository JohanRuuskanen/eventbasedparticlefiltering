
function eventSampling!(zout::AbstractArray{T,1}, yhout::AbstractArray{T,1},
     y::AbstractArray{T,1}, z::AbstractArray{T,1}, xh::AbstractArray{T,1},
     sys::sys_params, par::pf_params, δ, M) where T <: Real

    if par.eventKernel == "SOD"
        zout .= z
    elseif par.eventKernel == "MBT"
        zout .= sys.C * sys.A * xh
    else
        error("No such event kernel is implemented!")
    end

    if norm(z - y) >= δ
        γ = 1
        zout .= y
        yhout .= NaN # repeat(y, M)
    else
        γ = 0
        # Discretisize the uniform distribution, currently only supports dim(y) = 1
        yhout .= vcat(range(zout .- δ, stop=(zout .+ δ), length=M)...)
    end

    return γ

end

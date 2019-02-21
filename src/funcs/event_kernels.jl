
function eventSampling(y::Array{Float64,1}, z::Array{Float64,1},
     xh::Array{Float64,1}, sys::sys_params, par::pf_params, δ, M)

    if par.eventKernel == "SOD"
        z = z
    elseif par.eventKernel == "MBT"
        z = sys.C * sys.A * xh
    else
        error("No such event kernel is implemented!")
    end

    if norm(z - y) >= δ
        γ = 1
        z = y
    else
        γ = 0
        # Discretisize the uniform distribution, currently only supports dim(y) = 1
        yh = vcat(range(z .- δ, stop=(z .+ δ), length=M)...)
    end

    return z, γ, yh

end

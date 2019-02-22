
function propagation_bootstrap!(X::AbstractArray{T,2}, Xr::AbstractArray{T,2},
    sys::sys_params) where T <: Real
    N, M = size(Xr)
    distribution = MvNormal(zeros(M), sys.Q)
    rvec = Array{Float64,1}(undef, M)
    for i = 1:N
        # X[i,:] = sys.A*Xr[i, :] + rand!(distribution, rvec)
        rand!(distribution, rvec)
        @views mul!(X[i,:], sys.A, Xr[i, :])
        @inbounds X[i, :] .+= rvec
    end
    return X
end

function propagation_locallyOptimal(Xr::AbstractArray{T,2}, z::AbstractArray{T,1},
    sys::sys_params, yh, Vn, γ) where T <: Real

    JP_m(x) = [sys.A * x, sys.C * sys.A * x]
    JP_s(P) = [[sys.Q] [sys.Q*sys.C'];
                [sys.C*sys.Q] [sys.C * sys.Q * sys.C' + P]]

    N = size(Xr, 1)
    M = size(yh, 1)
    X = zeros(size(Xr))
    q_list = Array{Distribution}(undef, N)

    if γ == 1
        for i = 1:N

            μ = JP_m(Xr[i, :])
            Σ = JP_s(sys.R)

            P = fix_sym(Σ[1,1] - Σ[1, 2]*inv(Σ[2, 2])*Σ[1,2]')

            q_list[i] = MvNormal(μ[1] + Σ[1,2]*inv(Σ[2,2])*(z - μ[2]), P)
            X[i, :] = rand(q_list[i])
        end
    else
        for i = 1:N

            μ = JP_m(Xr[i, :])
            Σ = JP_s(sys.R .+ Vn)

            wh = zeros(M)
            for j = 1:M
                wh[j] = pdf(MvNormal(μ[2], Σ[2, 2]), yh[j, :])
            end

            if sum(wh) > 0
                wh = wh ./ sum(wh)
            else
                println("Bad conditioned weights for Mixture Gaussian; resetting to uniform")
                wh = 1 / M * ones(M)
            end

            P = fix_sym(Σ[1,1] .- Σ[1, 2]*inv(Σ[2, 2])*Σ[1,2]')
            μ_func(yh) = μ[1] .+ Σ[1,2]*inv(Σ[2,2])*(yh .- μ[2])

            MD = MixtureModel(map(y -> MvNormal([μ_func(y)...], P), yh), wh)

            q_list[i] = MD
            X[i, :] = rand(q_list[i])
        end
    end

    return X, q_list

end

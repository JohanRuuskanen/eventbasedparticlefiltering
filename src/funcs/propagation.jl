
function propagation_bootstrap(Xr::Array{Float64,2}, sys::sys_params)
    N, M = size(Xr)
    X = zeros(size(Xr))
    for i = 1:N
        X[i, :] = sys.A*Xr[i, :] + rand(MvNormal(zeros(M), sys.Q))
    end
    return X
end

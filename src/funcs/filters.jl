function kalman_filter(y, A, C, Q, R, sys; T=Float64, x0=Array{Float64,1}())

    xp = zeros(T, sys.nx, 1)
    Pp = zeros(T, sys.nx, sys.nx)

    x = zeros(T, sys.nx, sys.t_end)
    P = zeros(T, sys.nx, sys.nx, sys.t_end)

	if !isempty(x0)
		x[:, 1] .= x0
	end
    P[:,:,1] = Q

    for k = 2:sys.t_end
        xp = A*x[:, k-1]
        Pp = A*P[:, :, k-1]*A' + Q

        e = y[:, k] - C*xp
        S = R + C*Pp*C'
        K = Pp*C' / S
        x[:, k] = xp + K*e
        P[:,:,k] = (I - K*C)*Pp*(I - K*C)' + K*R*K'
    end

    return x, P
end



"""
Time periodic particle filters
"""
function bpf(y, sys, par)
    """
    Bootstrap particle filter, run for each sample y
    """

    nx = size(sys.A, 1)
    ny = size(sys.C, 1)

    X = zeros(par.N, nx, sys.T)
    W = zeros(par.N, sys.T)

    X[:, :, 1] = rand(Normal(0, 1), par.N, nx)
    W[:, 1] = 1/par.N .* ones(par.N, 1)

    idx = collect(1:par.N)
    Xr = X[:, :, 1]

    for k = 2:sys.T

        # Resample using systematic resampling
        idx = collect(1:par.N)
        wc = cumsum(W[:, k-1])
        u = (([0:(par.N-1)] + rand()) / par.N)[1]
        c = 1
        for i = 1:par.N
            while wc[c] < u[i]
                c = c + 1
            end
            idx[i] = c
        end

        # Resample
        for i = 1:par.N
            Xr[i, :] = X[idx[i], :, k-1]
        end

        # Propagate
        for i = 1:par.N
            X[i, :, k] = sys.A * Xr[i, :] + rand(MvNormal(zeros(nx), sys.Q))
        end

        # Weight
        for i = 1:par.N
            W[i, k] = pdf(MvNormal(sys.C * X[i, :, k], sys.R), y[:, k])
        end
        W[:, k] = W[:, k] ./ sum(W[:, k])

    end

    return X, W
end

function apf(y, sys, par)
    """
    Auxiliary particle filter
    """

    nx = size(sys.A, 1)
    ny = size(sys.C, 1)

    X = zeros(par.N, nx, sys.T)
    W = zeros(par.N, sys.T)
	V = zeros(par.N, sys.T)

    X[:, :, 1] = rand(Normal(0, 1), par.N, nx)
    W[:, 1] = 1/par.N .* ones(par.N, 1)
	V[:, 1] = 1/par.N .* ones(par.N, 1)

    idx = collect(1:par.N)

	q_list = Array{Distribution}(par.N)
	q_aux_list = Array{Distribution}(par.N)

	Xr = X[:, :, 1]

    for k = 2:sys.T

		# Calculate the auxiliary weights
	for i = 1:par.N
            μ = sys.C * sys.A * X[i, :, k-1]
            Σ = sys.C * sys.Q * sys.C' + sys.R
	    q_aux_list[i] = MvNormal(μ, Σ)
	    V[i, k-1] = W[i, k-1] * pdf(q_aux_list[i], y[:,k])
	end
	V[:, k-1] = V[:, k-1] ./ sum(V[:, k-1])

        # Resample using systematic resampling
        #idx = rand(Categorical(W[:, k-1]), N)
        wc = cumsum(V[:, k-1])
        u = (([0:(par.N-1)] + rand()) / par.N)[1]
        c = 1
        for i = 1:par.N
            while wc[c] < u[i]
                c = c + 1
            end
            idx[i] = c
        end

        for i = 1:par.N
            Xr[i, :] = X[idx[i], :, k-1]
        end
	q_aux_list = q_aux_list[idx]

        # Propagate
        for i = 1:par.N
            S = sys.C * sys.Q * sys.C' + sys.R

            μ = sys.A*Xr[i, :] + sys.Q*sys.C' * inv(S) * (y[k] - sys.C*sys.A*Xr[i, :])
            Σ = sys.Q - sys.Q*sys.C' * inv(S) * sys.C*sys.Q

            q_list[i] = MvNormal(μ, Σ)
            X[i, :, k] = rand(q_list[i])
        end

        # Weight
        for i = 1:par.N
            W[i, k] = 	pdf(MvNormal(sys.C*X[i, :, k], sys.R), y[:, k]) *
			pdf(MvNormal(sys.A*Xr[i, :], sys.Q), X[i, :, k]) /
                        (pdf(q_list[i], X[i, :, k]) * pdf(q_aux_list[i], y[:, k]))
        end
        W[:, k] = W[:, k] ./ sum(W[:, k])

    end

    return X, W
end

function kalman_filter(y, sys)

    # Extract parameters
    A = sys.A
    C = sys.C
    Q = sys.Q
    R = sys.R

    T = sys.T
    
    nx = size(A, 1)
    ny = size(C, 1)

    xp = zeros(nx, 1)
    Pp = zeros(nx, nx)

    x = zeros(nx, T)
    P = zeros(nx, nx, T)
    P[:,:,1] = eye(nx)

    for k = 2:T
        xp = A*x[:, k-1]
        Pp = A*P[:, :, k-1]*A' + Q

        e = y[:, k] - C*xp
        S = R + C*Pp*C'
        K = Pp*C'*inv(S)
        x[:, k] = xp + K*e
        P[:,:,k] = (eye(nx) - K*C)*Pp*(eye(nx) - K*C)' + K*R*K'
    end

    return x, P

end


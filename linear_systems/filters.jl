

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

    N_T = par.N / 2
    N_eff = zeros(T)

    for k = 2:sys.T

        N_eff[k-1] = 1/(sum(W[:, k-1].^2))
        println(N_eff[k-1])
        if N_eff[k-1] < N_T
            # Resample using systematic resampling
            idx = collect(1:par.N)
            wc = cumsum(W[:, k-1])
            u = (([0:(par.N-1)] .+ rand()) / par.N)[1]
            c = 1
            for i = 1:par.N
                while wc[c] < u[i]
                    c = c + 1
                end
                idx[i] = c
            end

            X[:, :, k-1] = X[idx, :, k-1]
            W[:, k-1] = 1/par.N .* ones(par.N, 1)
        end

        # Propagate
        for i = 1:par.N
            X[i, :, k] = sys.A * X[i, :, k-1] + rand(MvNormal(zeros(nx), sys.Q))
        end

        # Weight
        for i = 1:par.N
            W[i, k] = pdf(MvNormal(sys.C * X[i, :, k], sys.R), y[:, k]) * W[i, k-1]
        end
        W[:, k] = W[:, k] ./ sum(W[:, k])

    end

    return X, W, N_eff
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

	q_list = Array{Distribution}(undef, par.N)
	q_aux_list = Array{Distribution}(undef, par.N)

	Xr = X[:, :, 1]
    N_T = par.N / 2
    N_eff = zeros(T)

    resample = false

    for k = 2:sys.T

        # Calculate the auxiliary weights
    	for i = 1:par.N
                μ = sys.C * sys.A * X[i, :, k-1]
                Σ = sys.C * sys.Q * sys.C' + sys.R
    	    q_aux_list[i] = MvNormal(μ, Σ)
    	    V[i, k-1] = W[i, k-1] * pdf(q_aux_list[i], y[:,k])
    	end
    	V[:, k-1] = V[:, k-1] ./ sum(V[:, k-1])

        N_eff[k-1] = 1 ./ (sum(V[:, k-1].^2))
        println(N_eff[k-1])

        if N_eff[k-1] < N_T
            # Resample using systematic resampling
            wc = cumsum(V[:, k-1])
            u = (([0:(par.N-1)] .+ rand()) / par.N)[1]
            c = 1
            for i = 1:par.N
                while wc[c] < u[i]
                    c = c + 1
                end
                idx[i] = c
            end

    	    q_aux_list = q_aux_list[idx]
            X[:, :, k-1] = X[idx, :, k-1]
            W[:, k-1] = 1/par.N .* ones(par.N, 1)
            resample = true
        else
            resample = false
        end


        # Propagate
        for i = 1:par.N
            S = sys.C * sys.Q * sys.C' + sys.R

            μ = sys.A*X[i, :, k-1] + sys.Q*sys.C' * inv(S) * (y[k] .- sys.C*sys.A*X[i, :, k-1])
            Σ = sys.Q - sys.Q*sys.C' * inv(S) * sys.C*sys.Q

            q_list[i] = MvNormal(μ, Σ)
            X[i, :, k] = rand(q_list[i])
        end

        # Weight
        for i = 1:par.N
            if resample == true
                W[i, k] =  pdf(MvNormal(sys.C*X[i, :, k], sys.R), y[:, k]) *
    			             pdf(MvNormal(sys.A*X[i, :, k-1], sys.Q), X[i, :, k]) /
                            (pdf(q_list[i], X[i, :, k]) * pdf(q_aux_list[i], y[:, k]))
            elseif resample == false
                W[i, k] =  W[i, k-1] * pdf(MvNormal(sys.C*X[i, :, k], sys.R), y[:, k]) *
    			             pdf(MvNormal(sys.A*X[i, :, k-1], sys.Q), X[i, :, k]) /
                            (pdf(q_list[i], X[i, :, k]))
            end
        end
        W[:, k] = W[:, k] ./ sum(W[:, k])

    end

    return X, W, N_eff
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

function FFBSi(X, W, sys, M)

    # Extract parameters
    A = sys.A
    C = sys.C
    Q = sys.Q
    R = sys.R

    T = sys.T
    N = size(W, 1)

    nx = size(sys.A, 1)
    ny = size(sys.C, 1)

    Bs = zeros(Int64, M, T)
    Xs = zeros(M, size(X, 2), T)

    Bs[:, T] = rand(Categorical(W[:, T]), M)
    Xs[:, :, T] = X[Bs[:, T], :, T]

    for t = (T-1):-1:1
        println(t)
        for j = 1:M
            w = zeros(N)
            for i = 1:N
                w[i] = W[i, t] .* pdf(MvNormal(A*X[i, :, t]), Xs[j, :, t+1])
            end
            w /= sum(w)

            Bs[j, t] = rand(Categorical(w))
            Xs[j, :, t] = X[Bs[j, t], :, t]
        end
    end

    return Xs
end



"""
Time periodic particle filters
"""
function bpf(y, sys, N)
    """
    Bootstrap particle filter, run for each sample y
    """
    f = sys.f
    h = sys.h

    nx = sys.nd[1]
    ny = sys.nd[2]

    T = sys.T

    X = zeros(N, nx, T)
    W = zeros(N, T)

    X[:, :, 1] = rand(Normal(0, 10), N, nx)
    W[:, 1] = 1/N .* ones(N, 1)

    idx = collect(1:N)
    Xr = X[:, 1]

    for k = 2:T

        # Resample using systematic resampling
        idx = collect(1:N)
        wc = cumsum(W[:, k-1])
        u = (([0:(N-1)] + rand()) / N)[1]
        c = 1
        for i = 1:N
            while wc[c] < u[i]
                c = c + 1
            end
            idx[i] = c
        end

        # Resample
        for i = 1:N
            Xr[i, :] = X[idx[i], :, k-1]
        end

        # Propagate
        for i = 1:N
            X[i, :, k] = f(Xr[i, :], k) + rand(sys.w)
        end

        # Weight
        for i = 1:N
            W[i, k] = pdf.(sys.v, y[k] - h(X[i, :, k], k)[1])
        end
        if sum(W[:, k]) > 0
            W[:, k] = W[:, k] ./ sum(W[:, k])
        else
            println("Warning BPF: restting weights to uniform")
            W[:, k] = 1/N .* ones(N, 1)
        end
    end

    return X, W
end

function apf(y, sys, par)
    """
    Auxiliary particle filter
    """

    f = sys.f
    h = sys.h

    Q = std(sys.w)
    R = std(sys.v)

    Qh = std(sys.w)*2
    Rh = std(sys.v)*2

    nx = sys.nd[1]
    ny = sys.nd[2]

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
            p = f(X[i, :, k-1], k)

            μ = p.^2/20 + Q/20
            Σ = p.^2*Q / 100 + Q^2/200 + R
    	    q_aux_list[i] = MvNormal(reshape(μ, 1), Σ)
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

        for i = 1:par.N
            p = f(Xr[i, :], k)

            μ = [p, p.^2/20 + Q/20]
            Σ = [Q p*Q/10;
                p*Q/10 p.^2*Q / 100 + Q^2/200 + R]

            q_list[i] = MvNormal(μ[1] + Σ[1,2]*inv(Σ[2,2])*(y[:,k]-μ[2]),
                                Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2]')
            X[i, :, k] = rand(q_list[i])
        end

        # Weight
        for i = 1:par.N
            W[i, k] =   pdf.(sys.v, y[k] - h(X[i, :, k], k)[1]) *
                        pdf.(sys.w, X[i, :, k][1] - f(Xr[i, :], k)[1]) /
                        (pdf(q_list[i], X[i, :, k]) * pdf(q_aux_list[i], y[:, k]))
        end
        if sum(W[:, k]) > 0
            W[:, k] = W[:, k] ./ sum(W[:, k])
        else
            println("Warning APF: restting weights to uniform")
            W[:, k] = 1/N .* ones(N, 1)
        end
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

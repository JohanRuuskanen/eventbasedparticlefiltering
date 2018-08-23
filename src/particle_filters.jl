

"""
Time periodic particle filters
"""
function bpf(y, sys, par)
    """
    Bootstrap particle filter, run for each sample y
    """

    X = zeros(par.N, sys.nd[1], sys.T)
    W = zeros(par.N, sys.T)

    X[:, :, 1] = rand(Normal(0, 1), N)
    W[:, 1] = 1/par.N .* ones(par.N, 1)

    idx = collect(1:N)
    Xr = X[:, :, 1]

    for k = 2:sys.T

        # Resample using systematic resampling
        #idx = rand(Categorical(W[:, k]), N)
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
            X[i, :, k] = rand(sys.f(Xr[i, :], k))
        end

        # Weight
        for i = 1:par.N
            W[i, k] = pdf(sys.h(X[i, :, k], k), y[:, k])
        end
        W[:, k] = W[:, k] ./ sum(W[:, k])



    end

    return X, W
end

function sir(y, sys, par)
    """
    Sequential importance resampling particle filter
    """

    X = zeros(par.N, sys.nd[1], sys.T)
    W = zeros(par.N, sys.T)

    X[:, :, 1] = rand(Normal(0, 1), N)
    W[:, 1] = 1/par.N .* ones(par.N, 1)

    Q = var(sys.f([0], [0]))
    R = var(sys.h([0], [0]))

    idx = collect(1:N)
    q_list = Array{Distribution}(N)

    Xr = X[:, :, 1]
    Wr = W[:, 1]

    for k = 2:sys.T

        # Resample using systematic resampling
        #idx = rand(Categorical(W[:, k-1]), N)

        wc = cumsum(W[:, k-1])
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
            Wr[i, :] = W[idx[i], k-1]
        end

        # Propagate
        for i = 1:par.N

            μ = [mean(sys.f(Xr[i, :], k)), mean(sys.f(Xr[i, :], k)).^2/20 + Q/20]
            Σ = [Q mean(sys.f(Xr[i, :], k)).*Q/10;
                    mean(sys.f(Xr[i, :], k)).*Q/10 mean(sys.f(Xr[i, :], k)).^2.*Q/100 + Q.^2/200 + R]

            Mk = μ[1] + Σ[1, 2]/(Σ[2,2])*(y[k] - μ[2])
            Sk = Σ[1, 1] - Σ[1, 2]/(Σ[2,2])*(Σ[1, 2]')

            q = MvNormal(Mk, 1.2*Sk)

            X[i, :, k] = rand(q)
            q_list[i] = q
        end

        # Weight
        Z = 0
        for i = 1:par.N
            W[i, k] = 	pdf(sys.h(X[i, :, k], k), y[:, k]) *
						pdf(sys.f(Xr[i, :], k), X[i, :, k]) /
                        pdf(q_list[i], X[i, :, k])
        end

        W[:, k] = W[:, k] ./ sum(W[:, k])

    end

    return X, W
end

function apf(y, sys, par)
    """
    Auxiliary particle filter
    """

    X = zeros(par.N, sys.nd[1], sys.T)
    W = zeros(par.N, sys.T)
	V = zeros(par.N, sys.T)

    X[:, :, 1] = rand(Normal(0, 1), N)
    W[:, 1] = 1/par.N .* ones(par.N, 1)
	V[:, 1] = 1/par.N .* ones(par.N, 1)

    Q = var(sys.f([0], [0]))
    R = var(sys.h([0], [0]))

    idx = collect(1:N)

	mu_list = Array{Any}(N)
	sig_list = Array{Any}(N)
	q_list = Array{Distribution}(N)
	q_aux_list = Array{Distribution}(N)

	scale = 1.2
	Xr = X[:, :, 1]

    for k = 2:sys.T

		# Calculate the auxiliary weights
		for i = 1:par.N
			μ = mean(sys.f(X[i, :, k-1], k)).^2/20 + Q/20
			Σ = mean(sys.f(X[i, :, k-1], k)).^2.*Q/100 + Q.^2/200 + R

			q_aux_list[i] = MvNormal(μ, scale*Σ)
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
			μ = [mean(sys.f(Xr[i, :], k)), mean(sys.f(Xr[i, :], k)).^2/20 + Q/20]
            Σ = [Q mean(sys.f(Xr[i, :], k)).*Q/10;
                    mean(sys.f(Xr[i, :], k)).*Q/10 mean(sys.f(Xr[i, :], k)).^2.*Q/100 + Q.^2/200 + R]

			Mk = μ[1] + Σ[1, 2]/(Σ[2,2])*(y[k] - μ[2])
			Sk = Σ[1, 1] - Σ[1, 2]/(Σ[2,2])*(Σ[1, 2]')

            q_list[i] = MvNormal(Mk, Sk)
            X[i, :, k] = rand(q_list[i])
        end

        # Weight
        for i = 1:par.N
            W[i, k] = 	pdf(sys.h(X[i, :, k], k), y[:, k]) *
						pdf(sys.f(Xr[i, :], k), X[i, :, k]) /
                        (pdf(q_list[i], X[i, :, k]) *
						pdf(q_aux_list[i], y[:, k]))

        end
        W[:, k] = W[:, k] ./ sum(W[:, k])

    end

    return X, W
end

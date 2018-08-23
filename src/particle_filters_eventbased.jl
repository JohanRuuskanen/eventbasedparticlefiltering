
"""
Event driven particle filters using the MBT kernel
"""
function ebpf_naive(y, sys, par, δ)
    """
	Bootstrap particle filter, update on new measurement only
    """

    X = zeros(par.N, sys.nd[1], sys.T)
    W = zeros(par.N, sys.T)

    xh = zeros(sys.nd[1], sys.T)
    yh = zeros(sys.nd[2], sys.T)
    Z = zeros(1, sys.T)
    Γ = zeros(1, sys.T)

    X[:, :, 1] = rand(sys.f(zeros(sys.nd[1]), 0), par.N)
    W[:, 1] = 1/par.N .* ones(par.N, 1)

    for k = 1:nd[1]
        xh[k, :] = sum(diag(W'*X[:, k, :]), 2)
    end

    Xr = X[:, :, 1]

    for k = 2:sys.T

        xh[:, k], yh[:, k], Z[:, k], Γ[:, k] = EventKer_MBT(xh[:, k-1], Z[:, k-1], y[:, k], k, δ, sys)

        # Propagate
        for i = 1:par.N
            X[i, :, k] = rand(sys.f(Xr[i, :], k))
        end

        # With no event generated, the particles will only be propagated, not
        # resampled. Weights are kept constant.
        if Γ[k] == 0
            Xr = X[:, :, k]
            W[:, k] = W[:, k-1]
            continue
        end

        # If event was generated, update weights and resample particles and
        # update the estimate at this event.

        # Weight
        for i = 1:par.N
            W[i, k] = pdf(sys.h(X[i, :, k], k), y[:, k])
        end

        # If weights are too small, set uniform weight
        if sum(W[:, k]) > 0
            W[:, k] = W[:, k] ./ sum(W[:, k])
        else
            W[:, k] = 1/par.N * ones(par.N, 1)
        end


        for i = 1:nd[1]
            xh[i, k] = sum(diag(W'*X[:, i, :]), 2)[k]
        end

        # Resample using systematic resampling
        #idx = rand(Categorical(W[:, k]), N)

        idx = collect(1:par.N)
        wc = cumsum(W[:, k])
        u = (([0:(par.N-1)] + rand()) / par.N)[1]
        c = 1
        for i = 1:par.N
            while wc[c] < u[i]
                c = c + 1
            end
            idx[i] = c
        end

        for i = 1:par.N
            Xr[i, :] = X[idx[i], :, k]
        end

    end

    return X, W, xh, yh, Z, Γ
end

function ebpf_usez(y, sys, par, δ)
    """
    event driven bootstrap particle filter that utilizes z every time,
    for the MBT kernel
    """

    X = zeros(par.N, sys.nd[1], sys.T)
    W = zeros(par.N, sys.T)

    xh = zeros(sys.nd[1], sys.T)
    yh = zeros(sys.nd[2], sys.T)
    Z = zeros(1, sys.T)
    Γ = zeros(1, sys.T)

    X[:, :, 1] = rand(sys.f(zeros(sys.nd[1]), 0), par.N)
    W[:, 1] = 1/par.N .* ones(par.N, 1)

    for k = 1:nd[1]
        xh[k, :] = mean(diag(W'*X[:, k, :]), 2)
    end

    Xr = X[:, :, 1]

    for k = 2:sys.T

        xh[:, k], yh[:, k], Z[:, k], Γ[:, k] = EventKer_MBT(xh[:, k-1], Z[:, k-1], y[:, k], k, δ, sys)

        # Propagate
        for i = 1:par.N
            X[i, :, k] = rand(sys.f(Xr[i, :], k))
        end

        # Weight
        if Γ[k] == 1
            for i = 1:par.N
                W[i, k] = pdf(sys.h(X[i, :, k], k), y[:, k])
            end
        else
            for i = 1:par.N
                if δ > 0

                    Uniform(yh[k] - δ, yh[k] + δ)

                    Dh = h(X[i, :, k], k)
                    D = Normal(mean(Dh)[1], var(Dh)[1])

                    W[i, k] = cdf(D, yh[k] + δ) - cdf(D, yh[k] - δ)
                else
                    println("Something weird happened")
                    W[i, k] = 1/N
                end
            end
        end

        if sum(W[:, k]) > 0
            W[:, k] = W[:, k] ./ sum(W[:, k])
        else
            # If weights are too small, we set uniform weights to the particles
            W[:, k] = 1/par.N * ones(par.N, 1)
        end

        if Γ[k] == 1
            for i = 1:nd[1]
                xh[i, k] = mean(diag(W'*X[:, i, :]), 2)[k]
            end
        end

        # Resample using systematic resampling
        #idx = rand(Categorical(W[:, k]), N)

        idx = collect(1:par.N)
        wc = cumsum(W[:, k])
        u = (([0:(par.N-1)] + rand()) / par.N)[1]
        c = 1
        for i = 1:par.N
            while wc[c] < u[i]
                c = c + 1
            end
            idx[i] = c
        end

        for i = 1:par.N
            Xr[i, :] = X[idx[i], :, k]
        end


    end

    return X, W, xh, yh, Z, Γ
end


function eapf(y, sys, par)
    """
    Event based auxiliary particle filter, update on new measurement only
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


function naive_filter(y, sys, δ)

    xh = zeros(sys.nd[1], sys.T)
    yh = zeros(sys.nd[2], sys.T)
    Z = zeros(1, sys.T)
    Γ = zeros(1, sys.T)

    for k = 2:sys.T
        xh[:, k], yh[:, k], Z[:,k], Γ[:,k] = EventKer_MBT(xh[:, k-1], Z[:, k-1], y[:, k], k, δ, sys)
    end

    return xh, yh, Z, Γ

end#

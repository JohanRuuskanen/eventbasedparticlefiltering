"""
Event driven particle filters
"""
function ebpf(y, sys, par, δ)
    """
    Event-based bootstrap particle filter
    """

    f = sys.f
    h = sys.h

    w = sys.w
    v = sys.v

    nx = sys.nd[1]
    ny = sys.nd[2]

    N = par.N
    T = sys.T

    X = zeros(N, nx, T)
    W = zeros(N, T)
    Z = zeros(ny, T)
    Γ = zeros(T)

    X[:, :, 1] = rand(Normal(0, 10), N, nx)
    W[:, 1] = 1/N .* ones(N, 1)

    Neff = zeros(T)
    res = zeros(T)
    fail = zeros(T)

    N_T = N / 2

    idx = collect(1:N)
    Xr = X[:, 1]
    Wr = W[:, 1]

    for k = 2:T

        # Run event kernel, SOD
        if norm(Z[:, k-1] - y[:, k]) >= δ
            Γ[k] = 1
            Z[:, k] = y[:, k]
        else
            Γ[k] = 0
            Z[:, k] = Z[:, k-1]
        end

        Neff[k] = 1/sum(W[:, k-1].^2)
        if Neff[k] < N_T
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
            
            Xr = X[idx, :, k-1]
            Wr = 1/N .* ones(N, 1)
            res[k] = 1
        else
            Xr = X[:, :, k-1]
            Wr = W[:, k-1]
        end

        # Propagate
        for i = 1:N
            X[i, :, k] = f(Xr[i, :], k) + rand(sys.w)
        end

        # Weight
        if Γ[k] == 1
            for i = 1:N
                W[i, k] = log(Wr[i]) + log(pdf.(sys.v, y[k] - h(X[i, :, k], k)[1]))
            end
        elseif Γ[k] == 0
            for i = 1:N
                D = Normal(h(X[i, :, k], k)[1], std(v))
                W[i, k] = log(Wr[i]) + log(cdf(D, Z[k] + δ) - cdf(D, Z[k] - δ))
            end
        end

        W_max = maximum(W[:, k])
        if W_max > -Inf
            W_tmp = exp.(W[:, k] - W_max*ones(N, 1))
            W[:, k] = W_tmp ./ sum(W_tmp)
        else
            println("Warning BPF: restting weights to uniform")
            W[:, k] = 1/N .* ones(N, 1)
            fail[k] = 1
            Neff[k] = 0
        end

    end
    return X, W, Z, Γ, Neff, res, fail
end

function eapf(y, sys, par, δ)
    """
    Event-based auxiliary particle filter
    """
    f = sys.f
    h = sys.h
    w = sys.w
    v = sys.v

    Q = var(w)
    R = var(v)

    nx = sys.nd[1]
    ny = sys.nd[2]

    N = par.N

    X = zeros(par.N, nx, sys.T)
    W = zeros(par.N, sys.T)
	V = zeros(par.N, sys.T)

    Z = zeros(ny, T)
    Γ = zeros(T)

    X[:, :, 1] = rand(Normal(0, 1), par.N, nx)
    W[:, 1] = 1/par.N .* ones(par.N, 1)
	V[:, 1] = 1/par.N .* ones(par.N, 1)

    idx = collect(1:par.N)

	q_list = Array{Distribution}(par.N)
	q_aux_list = Array{Distribution}(par.N)

	Xr = X[:, :, 1]
    Wr = W[:, 1]

    Neff = zeros(T)
    res = zeros(T)
    fail = zeros(T)

    N_T = N/2

    # === For approximating the uniform distribution
    # number of approximation points and their spacing
    M = 20 #ceil(2 * δ) + 1
    L = 2*δ / (M-1)

    Vn = L / sqrt(2)
    yh = zeros(M)
    # ===

    JP_mu(x, k) = [f(x, k), f(x, k).^2/20 + Q/20]
    JP_sig(x, k, S) = [Q f(x, k).*Q/10;
                    f(x, k).*Q/10 (f(x, k).^2*Q/100 + Q.^2/200 + S)]

    for k = 2:T

        # Run event kernel, SOD
        if norm(Z[:, k-1] - y[:, k]) >= δ
            Γ[k] = 1
            Z[:, k] = y[:, k]
        else
            Γ[k] = 0
            Z[:, k] = Z[:, k-1]

            # Discretisize the uniform distribution, currently only supports
            # dim(y) = 1
            yh = vcat(linspace(Z[:, k]- δ, Z[:, k] + δ, M)...)
        end

        # Calculate auxiliary weights
        if Γ[k] == 1
            for i = 1:par.N
                μ = JP_mu(X[i, :, k-1], k)
                Σ = JP_sig(X[i, :, k-1], k, R)

                q_aux_list[i] = MvNormal(μ[2], sqrt(Σ[2,2]))
                V[i, k-1] = log(W[i, k-1]) + log(pdf(q_aux_list[i], y[:,k]))
        	end
        else
            for i = 1:par.N
                μ = JP_mu(X[i, :, k-1], k)
                Σ = JP_sig(X[i, :, k-1], k, R + Vn)

                q_aux_list[i] = MvNormal(μ[2], sqrt(Σ[2,2]))
                predLh = 0
                for j = 1:M
                    predLh += pdf(q_aux_list[i], yh[j, :])
                end
                V[i, k-1] = log(W[i, k-1]) + log(predLh)
            end
        end
        V_max = maximum(V[:, k-1])
        if V_max > -Inf
            V_tmp = exp.(V[:, k-1] - V_max*ones(N, 1))
            V[:, k-1] = V_tmp ./ sum(V_tmp)
        else
            println("Warning APF: setting auxiliary weights to W(k-1)")
            V[:, k-1] = W[:, k-1]
            fail[k] = 1
            Neff[k] = 0
        end

        Neff[k] = 1/sum(V[:, k-1].^2)
        if Neff[k] < N_T 
            # Resample using systematic resampling
            idx = collect(1:N)
            wc = cumsum(V[:, k-1])
            u = (([0:(N-1)] + rand()) / N)[1]
            c = 1
            for i = 1:N
                while wc[c] < u[i]
                    c = c + 1
                end
                idx[i] = c
            end

            Xr = X[idx, :, k-1]
            Wr = 1/N .* ones(N, 1)
            q_aux_list = q_aux_list[idx]
            res[k] = 1
        else
            Xr = X[:, :, k-1]
            Wr = W[:, k-1]
        end


        # Propagate
        if Γ[k] == 1
            for i = 1:N
                μ = JP_mu(Xr[i, :], k)
                Σ = JP_sig(Xr[i, :], k, R)
                Σ = fix_sym(Σ)

                q_list[i] = MvNormal(μ[1] + Σ[1,2]*inv(Σ[2,2])*(y[:,k] - μ[2]),
                                     sqrt(Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2]'))
                X[i, :, k] = rand(q_list[i])
            end
        else
            for i = 1:par.N
                μ = JP_mu(Xr[i, :], k)
                Σ = JP_sig(Xr[i, :], k, R + Vn)
                Σ = fix_sym(Σ)

                μ_func(x) = μ[1] + Σ[1,2]*inv(Σ[2,2])*(x - μ[2])
                S = sqrt(Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2])

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

                MD = MixtureModel(map(y -> MvNormal([μ_func(y)...], S), yh), wh)

                q_list[i] = MD
                X[i, :, k] = rand(q_list[i])
            end
        end

        # Weight
        if Γ[k] == 1
            for i = 1:par.N
                if res[k] == 1
                    W[i, k] = log(Wr[i]) + log(pdf(v, Z[:,k] - h(X[i, :, k], k))[1]) +
                                log(pdf(w, X[i, :, k] - f(Xr[i, :], k))[1]) -
                                log(pdf(q_list[i], X[i, :, k])) -
                                log(pdf(q_aux_list[i], Z[:, k]))
                else
                    W[i, k] = log(Wr[i]) + log(pdf(v, Z[:,k] - h(X[i, :, k], k))[1]) +
                                log(pdf(w, X[i, :, k] - f(Xr[i, :], k))[1]) -
                                log(pdf(q_list[i], X[i, :, k]))
                end
            end
        else
            for i = 1:par.N
                # Likelihood
                D = Normal(h(X[i, :, k], k)[1], sqrt(R))
                py = cdf(D, Z[k] + δ) - cdf(D, Z[k] - δ)

                # Propagation
                px = pdf(w, X[i, :, k] - f(Xr[i, :], k))[1]

                # proposal distribution
                q = pdf(q_list[i], X[i, :, k])
                
                if res[k] == 1
                    # Predictive likelihood
                    pL = 0
                    for j = 1:M
                        pL += pdf(q_aux_list[i], yh[j, :])
                    end

                    W[i, k] = log(Wr[i]) + log(py) + log(px) - log(pL) - log(q)
                else
                    W[i, k] = log(Wr[i]) + log(py) + log(px) - log(q)
                end


            end
        end

        W_max = maximum(W[:, k])
        if W_max > -Inf
            W_tmp = exp.(W[:, k] - W_max*ones(N, 1))
            W[:, k] = W_tmp ./ sum(W_tmp)
        else
            println("Bad conditioned weights for EAPF! Resetting to uniform")
            W[:, k] = 1/par.N * ones(par.N, 1)
            fail[k] = 1
            Neff[k] = 0
        end


    end

    return X, W, Z, Γ, Neff, res, fail
end

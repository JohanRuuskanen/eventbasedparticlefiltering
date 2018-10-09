"""
Event driven particle filters using the MBT kernel
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

    idx = collect(1:N)
    Xr = X[:, 1]

    for k = 2:T

        # Run event kernel, SOD
        if norm(Z[:, k-1] - y[:, k]) >= δ
            Γ[k] = 1
            Z[:, k] = y[:, k]
        else
            Γ[k] = 0
            Z[:, k] = Z[:, k-1]
        end

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
        if Γ[k] == 1
            for i = 1:N
                W[i, k] = pdf.(sys.v, y[k] - h(X[i, :, k], k)[1])
            end
        elseif Γ[k] == 0
            for i = 1:N
                D = Normal(h(X[i, :, k], k)[1], std(w))
                W[i, k] = cdf(D, Z[k] + δ) - cdf(D, Z[k] - δ)
            end
        end

        if sum(W[:, k]) > 0
            W[:, k] = W[:, k] ./ sum(W[:, k])
        else
            println("Warning BPF: restting weights to uniform")
            W[:, k] = 1/N .* ones(N, 1)
        end

    end
    return X, W, Z, Γ
end

function eapf(y, sys, par, δ)
    """
    Event-based auxiliary particle filter
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

    Z = zeros(ny, T)
    Γ = zeros(T)

    X[:, :, 1] = rand(Normal(0, 1), par.N, nx)
    W[:, 1] = 1/par.N .* ones(par.N, 1)
	V[:, 1] = 1/par.N .* ones(par.N, 1)

    idx = collect(1:par.N)

	q_list = Array{Distribution}(par.N)
	q_aux_list = Array{Distribution}(par.N)

	Xr = X[:, :, 1]

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

        	    q_aux_list[i] = MvNormal(μ[2], Σ[2,2])
        	    V[i, k-1] = W[i, k-1] * pdf(q_aux_list[i], y[:,k])
        	end
            V[:, k-1] = V[:, k-1] ./ sum(V[:, k-1])
        else
            for i = 1:par.N
                μ = JP_mu(X[i, :, k-1], k)
                Σ = JP_sig(X[i, :, k-1], k, R + Vn)

                q_aux_list[i] = MvNormal(μ[2], Σ[2,2])
                predLh = 0
                for j = 1:M
                    predLh += pdf(q_aux_list[i], yh[j, :])
                end
                V[i, k-1] = W[i, k-1] * predLh
            end
        end
	    V[:, k-1] = V[:, k-1] ./ sum(V[:, k-1])

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

        # Resample
        for i = 1:N
            Xr[i, :] = X[idx[i], :, k-1]
        end
        q_aux_list = q_aux_list[idx]

        # Propagate
        if Γ[k] == 1
            for i = 1:N
                μ = JP_mu(Xr[i, :], k)
                Σ = JP_sig(Xr[i, :], k, R)
                Σ = fix_sym(Σ)

                q_list[i] = MvNormal(μ[1] + Σ[1,2]*inv(Σ[2,2])*(y[:,k] - μ[2]),
                                    Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2]')
                X[i, :, k] = rand(q_list[i])
            end
        else
            for i = 1:par.N
                μ = JP_mu(Xr[i, :], k)
                Σ = JP_sig(Xr[i, :], k, R + Vn)
                Σ = fix_sym(Σ)

                μ_func(yh) = μ[1] + Σ[1,2]*inv(Σ[2,2])*(yh - μ[2])
                #Σ = Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2]'

                MD = MixtureModel(map(y -> MvNormal([μ_func(y)...],
                        Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2]), yh))

                q_list[i] = MD
                X[i, :, k] = rand(q_list[i])
            end
        end

        # Weight
        if Γ[k] == 1
            for i = 1:par.N
                W[i, k] = pdf(v, Z[:,k] - h(X[i, :, k], k))[1] *
                          pdf(w, X[i, :, k] - f(Xr[i, :], k))[1] /
                          (pdf(q_list[i], X[i, :, k]) * pdf(q_aux_list[i], Z[:, k]))
            end
        else
            for i = 1:par.N
                # Likelihood
                D = Normal(h(X[i, :, k], k)[1], R)
                py = cdf(D, Z[k] + δ) - cdf(D, Z[k] - δ)

                # Propagation
                px = pdf(w, X[i, :, k] - f(Xr[i, :], k))[1]

                # Predictive likelihood
                pL = 0
                for j = 1:M
                    pL += pdf(q_aux_list[i], yh[j, :])
                end
                # proposal distribution
                q = pdf(q_list[i], X[i, :, k])

                W[i, k] = py*px / (pL*q)

            end
        end


        if sum(W[:, k]) > 0
            W[:, k] = W[:, k] ./ sum(W[:, k])
        else
            println("Bad conditioned weights for EAPF! Resetting to uniform")
            W[:, k] = 1/par.N * ones(par.N, 1)
        end


    end

    return X, W, Z, Γ
end

"""
Event-based state estimator
"""
function ebse(y, sys, δ)

    # Extract parameters
    A = sys.A
    C = sys.C
    Q = sys.Q
    R = sys.R

    T = sys.T

    # === For approximating the uniform distribution
    # number of approximation points and their spacing
    M = 20 #ceil(2 * δ) + 1
    L = 2*δ / (M-1)

    Vn = L / sqrt(2)
    # ===

    nx = size(A, 1)
    ny = size(C, 1)

    xp = zeros(nx, T)
    Pp = zeros(nx, nx, T)
    yp = zeros(ny, T)
    Z = zeros(ny, T)
    Γ = zeros(T)

    x = zeros(nx, T)
    P = zeros(nx, nx, T)

    P[:,:,1] = eye(nx)

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

        # Predict
        xp = A*x[:, k-1]
        Pp = A*P[:, :, k-1]*A' + Q

        # Update
        S = inv(inv(Pp) + C'*inv(R)*C)

        if Γ[k] == 1
            x[:, k] = S*(inv(Pp)*xp + C'*inv(R)*Z[:, k])
            P[:, :, k] = S
        else
            θ = zeros(2, M)
            w = zeros(1, M)
            for i = 1:M
                θ[:, i] = S*(inv(Pp)*xp + C'*inv(R)*yh[i])
                w[:, i] = pdf(MvNormal(C*xp, C*Pp*C' + R + Vn), yh[i, :])
            end
            w /= sum(w)

            for i = 1:M
                x[:, k] += w[:, i] .* θ[:, i]
            end

            for i = 1:M
                P[:, :, k] += w[:, i] .* (S + (x[:, k] - θ[:, i])*(x[:, k] - θ[:, i])')
            end
        end
    end

    return x, P, Z, Γ

end

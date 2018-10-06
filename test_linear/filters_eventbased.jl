"""
Event driven particle filters using the MBT kernel
"""
function ebpf(y, sys, par, δ)
    """
    Event-based bootstrap particle filter
    """

    # Extract parameters
    A = sys.A
    C = sys.C
    Q = sys.Q
    R = sys.R

    T = sys.T
    N = par.N

    nx = size(A, 1)
    ny = size(C, 1)

    X = zeros(N, nx, T)
    W = zeros(N, T)

    Z = zeros(ny, T)
    Γ = zeros(T)

    X[:, :, 1] = rand(Normal(0, 1), N, nx)
    W[:, 1] = 1/N .* ones(N, 1)

    idx = collect(1:N)
    Xr = X[:, :, 1]

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
            X[i, :, k] = A*Xr[i, :] + rand(MvNormal(zeros(nx), Q))
        end

        # Weight
        if Γ[k] == 1
            for i = 1:N
                W[i, k] = pdf(MvNormal(C*X[i, :, k], R), y[:, k])
            end
        else
            for i = 1:N
                # There are no general cdf for multivariate distributions, this
                # only works if y is a scalar
                D = Normal((C*X[i, :, k])[1], R[1])
                W[i, k] = (cdf(D, Z[k] + δ) - cdf(D, Z[k] - δ))
            end
        end

        if sum(W[:, k]) > 0
            W[:, k] = W[:, k] ./ sum(W[:, k])
        else
            println("Bad conditioned weights for EBPF! Resetting to uniform")
            W[:, k] = 1/N * ones(N, 1)
        end


    end

    return X, W, Z, Γ
end

function eapf(y, sys, par, δ)
    """
    Event-based auxiliary particle filter
    """

    # Extract parameters
    A = sys.A
    C = sys.C
    Q = sys.Q
    R = sys.R

    T = sys.T
    N = par.N

    # === For approximating the uniform distribution
    # number of approximation points and their spacing
    M = 20 #ceil(2 * δ) + 1
    L = 2*δ / (M-1)

    Vn = L / sqrt(2)
    # ===

    nx = size(A, 1)
    ny = size(C, 1)

    X = zeros(N, nx, T)
    W = zeros(N, T)
    V = zeros(N, T)

    Z = zeros(ny, T)
    Γ = zeros(T)

    X[:, :, 1] = rand(Normal(0, 1), N, nx)
    W[:, 1] = 1/N .* ones(N, 1)
    V[:, 1] = 1/N .* ones(N, 1)

    idx = collect(1:N)

    q_list = Array{Distribution}(N)
    q_aux_list = Array{Distribution}(N)

    Xr = X[:, :, 1]

    yh = zeros(M)

    JP_m(x) = [A*x, C*A*x]
    JP_s(S) = [[Q] [Q*C'];
                [C*Q] [C*Q*C' + S]]
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
            for i = 1:N
                μ = JP_m(X[i, :, k-1])
                Σ = JP_s(R)

                q_aux_list[i] = MvNormal(μ[2], Σ[2,2])
        	    V[i, k-1] = W[i, k-1] * pdf(q_aux_list[i], Z[:,k])
            end
        else
            for i = 1:par.N
                μ = JP_m(X[i, :, k-1])
                Σ = JP_s(R + Vn)

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
                μ = JP_m(Xr[i, :])
                Σ = JP_s(R)
                
                S = fix_sym(Σ[1,1] - Σ[1, 2]*inv(Σ[2, 2])*Σ[1,2]')

                q_list[i] = MvNormal(μ[1] + Σ[1,2]*inv(Σ[2,2])*(Z[:,k] - μ[2]), S)
                X[i, :, k] = rand(q_list[i])
            end
        else
            for i = 1:par.N
                μ = JP_m(Xr[i, :])
                Σ = JP_s(R + Vn)

                S = fix_sym(Σ[1,1] - Σ[1, 2]*inv(Σ[2, 2])*Σ[1,2]')
                μ_func(yh) = μ[1] + Σ[1,2]*inv(Σ[2,2])*(yh - μ[2])

                MD = MixtureModel(map(y -> MvNormal([μ_func(y)...], S), yh))

                q_list[i] = MD
                X[i, :, k] = rand(q_list[i])
            end
        end

        # Weight
        if Γ[k] == 1
            for i = 1:par.N
                W[i, k] = pdf(MvNormal(C*X[i, :, k], R), Z[:,k]) *
                          pdf(MvNormal(A*Xr[i, :], Q), X[i, :, k]) /
                          (pdf(q_list[i], X[i, :, k]) * pdf(q_aux_list[i], Z[:, k]))
            end
        else
            for i = 1:par.N
                # Likelihood
                D = Normal((C*X[i, :, k])[1], R[1])
                py = cdf(D, Z[k] + δ) - cdf(D, Z[k] - δ)

                # Propagation
                px = pdf(MvNormal(A*Xr[i, :], Q), X[i, :, k])

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

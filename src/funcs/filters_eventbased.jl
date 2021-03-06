"""
Event driven particle filters using SOD
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

    M = 10

    nx = size(A, 1)
    ny = size(C, 1)

    X = zeros(N, nx, T)
    W = zeros(N, T)
    S = zeros(N, T)

    xh = zeros(nx, T)
    yh = zeros(M)

    Z = zeros(ny, T)
    Γ = zeros(T)

    Neff = zeros(T)
    fail = zeros(T)
    res = zeros(T)

    X[:, :, 1] = rand(Normal(0, 1), N, nx)
    W[:, 1] = 1/N .* ones(N, 1)
    S[:, 1] = collect(1:N)

    xh[:, 1] = W[:, 1]' * X[:,:,1]

    idx = collect(1:N)
    Xr = X[:, :, 1]
    Wr = W[:, 1]

    N_T = par.Nlim

    for k = 2:T

        xh[:, k-1] = W[:, k-1]' * X[:,:,k-1]
        Γ[k] = eventSampling!(view(Z, :, k), yh, view(y,:,k), view(Z, :, k-1),
            view(xh,:, k-1), sys, par, δ, M)

        Neff[k] = 1 ./ sum(W[:, k-1].^2)
        if Neff[k] <= N_T

            idx = resampling_systematic(W[:, k-1])

            Xr = X[idx, :, k-1]
            Wr = 1/N .* ones(N, 1)
            res[k] = 1
            S[:, k] = idx
        else
            Xr = X[:, :, k-1]
            Wr = W[:, k-1]
            S[:, k] = collect(1:N)
        end

        propagation_bootstrap!(view(X, :, :, k), Xr, sys)

        # Weight
        if Γ[k] == 1
            for i = 1:N
                W[i, k] = log(Wr[i]) + log(pdf(MvNormal(C*X[i, :, k], R), y[:, k]))
            end
        else
            for i = 1:N
                # There are no general cdf for multivariate distributions, this
                # only works if y is a scalar
                #D = Normal((C*X[i, :, k])[1], R[1])
                #W[i, k] = log(Wr[i]) + log((cdf(D, Z[k] + δ) - cdf(D, Z[k] - δ)))

                # constrained bayesian state estimation
                D = Normal((C*X[i, :, k])[1], R[1])
                #yh = C*(A*X[i, :, k] + rand(MvNormal(zeros(nx), Q))) + rand(MvNormal(zeros(ny), R))
                yp = C*X[i, :, k] #+ rand(MvNormal(zeros(ny), R))
                if norm(Z[:, k] -  yp) < δ
                    W[i, k] = log(Wr[i]) + log(pdf(D, yp[1]))
                else
                    W[i, k] = -Inf
                end

            end
        end

        if maximum(W[:, k]) > -Inf
            w_max = maximum(W[:, k])
            W_norm = sum(exp.(W[:, k] - w_max*ones(N, 1)))
            W[:, k] = exp.(W[:, k] - w_max*ones(N, 1)) / W_norm
        else
            println("Bad conditioned weights for EBPF! Resetting to uniform")
            W[:, k] = 1/N * ones(N, 1)
            fail[k] = 1
            Neff[k] = 0
        end


    end

    return output(X, W, S, Z, Γ, res, fail)
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
    #M = 20 #ceil(2 * δ) + 1
    #L = 2*δ / (M-1)
    #Vn = L / sqrt(2)

    M = 5 #ceil(2 * δ) + 1
    L = 2*δ / (M)
    Vn = L / 2
    # ===

    yh = zeros(M)

    nx = size(A, 1)
    ny = size(C, 1)

    X = zeros(N, nx, T)
    W = zeros(N, T)
    V = zeros(N, T)
    S = zeros(N, T)
    xh = zeros(nx, T)

    Z = zeros(ny, T)
    Γ = zeros(T)

    Neff = zeros(T)
    res = zeros(T)
    fail = zeros(T)

    X[:, :, 1] = rand(Normal(0, 1), N, nx)
    W[:, 1] = 1/N .* ones(N, 1)
    V[:, 1] = 1/N .* ones(N, 1)
    S[:, 1] = collect(1:N)
    xh[:, 1] = W[:, 1]' * X[:,:,1]

    idx = collect(1:N)

    q_list = Array{Distribution}(undef, N)
    q_aux_list = Array{Distribution}(undef, N)

    Wr = W[:, 1]
    Xr = X[:, :, 1]
    Wr = W[:, 1]

    N_T = par.Nlim
    yh = zeros(M)

    JP_m(x) = [A*x, C*A*x]
    JP_s(P) = [[Q] [Q*C'];
                [C*Q] [C*Q*C' + P]]
    for k = 2:T

        xh[:, k-1] = W[:, k-1]' * X[:,:,k-1]
        Γ[k] = eventSampling!(view(Z, :, k), yh, view(y,:,k), view(Z, :, k-1),
            view(xh,:, k-1), sys, par, δ, M)

        # Calculate auxiliary weights
        if Γ[k] == 1
            for i = 1:N
                μ = JP_m(X[i, :, k-1])
                Σ = JP_s(R)

                q_aux_list[i] = MvNormal(μ[2], Σ[2,2])
                V[i, k-1] = log(W[i, k-1]) + log(pdf(q_aux_list[i], Z[:,k]))
            end
        else
            for i = 1:par.N
                μ = JP_m(X[i, :, k-1])
                Σ = JP_s(R .+ Vn)

                q_aux_list[i] = MvNormal(μ[2], Σ[2,2])

                predLh = 0
                for j = 1:M
                    predLh += pdf(q_aux_list[i], yh[j, :])
                end
                predLh /= M

                V[i, k-1] = log(W[i, k-1]) + log(predLh)
            end
        end
        if maximum(V[:, k-1]) > -Inf
            V_max = maximum(V[:, k-1])
            V_norm = sum(exp.(V[:, k-1] - V_max*ones(N, 1)))
            V[:, k-1] = exp.(V[:, k-1] - V_max*ones(N, 1)) / V_norm
        else
            println("Bad conditioned weights for Auxiliary variable! Resetting to W(k-1)")
            V[:, k-1] = W[:, k-1]
            fail[k] = 1
        end

        Neff[k] = 1/sum(V[:, k-1].^2)
        if Neff[k] <= N_T

            idx = resampling_systematic(V[:, k-1])

            Xr = X[idx, :, k-1]
            Wr = 1/N * ones(N)
            q_aux_list = q_aux_list[idx]
            res[k] = 1
            S[:, k] = idx
        else
            Xr = X[:, :, k-1]
            Wr = W[:, k-1]
            S[:, k] = collect(1:N)
        end

        # Propagate
        X[:, :, k], q_list = propagation_locallyOptimal(Xr, Z[:, k],
            sys, yh, Vn, Γ[k])

        # Weight
        calculate_weights!(view(W, :, k), view(X, :, :, k), view(Z, :, k), Wr, Xr,
            yh, q_list, q_aux_list, res[k], Γ[k], δ, sys, par)

        if maximum(W[:, k]) > -Inf
            w_max = maximum(W[:, k])
            W_norm = sum(exp.(W[:, k] - w_max*ones(N, 1)))
            W[:, k] = exp.(W[:, k] - w_max*ones(N, 1)) / W_norm
        else
            println("Bad conditioned weights for EAPF! Resetting to uniform")
            W[:, k] = 1/par.N * ones(par.N, 1)
            fail[k] = 1
        end


    end

    return output(X, W, S, Z, Γ, res, fail)
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
    M = 6000 #ceil(2 * δ) + 1
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

    P[:,:,1] = Matrix{Float64}(I, nx, nx)

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
            yh = vcat(range(Z[:, k] .- δ, stop=(Z[:, k] .+ δ), length=M)...)
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
                w[:, i] .= pdf(MvNormal(C*xp, C*Pp*C' .+ R .+ Vn), yh[i, :])
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

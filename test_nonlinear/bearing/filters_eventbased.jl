

function test_mbt(x, y, sys, par, δ)

    f = sys.f
    h = sys.h

    T = sys.T

    Z = zeros(1, T)
    Γ = zeros(T)
    xh = zeros(1, T)

    xh[:, 1] = x[:, 1]

    for k = 2:T

        y_pred = h(f(xh[:, k-1], k), k)
        if norm(h(xh[:, k], k) - y[:, k]) >= δ
            Γ[k] = 1
            Z[:, k] = y[:, k]

        else
            Γ[k] = 0
            Z[:, k] = y_pred
        end

        if Γ[k] == 1
            xh[:, k] = x[:, k]
        else
            xh[:, k] = f(xh[:, k-1], k)
        end

    end

    return xh, Z, Γ

end

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


	X[:, :, 1] = rand(par.X0, N)
    W[:, 1] = 1/N .* ones(N, 1)

    Neff = zeros(T)
    res = zeros(T)
    fail = zeros(T)

    N_T = N / 2

    idx = collect(1:N)
    Xr = X[:, 1]
    Wr = W[:, 1]

    xh = zeros(nx, T)

    for k = 2:T

        # Run event kernel, SOD

        if sum(abs.(Z[:, k-1] - y[:, k]) .> δ) > 0
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
                W[i, k] = log(Wr[i]) + log(pdf(sys.v, y[:, k] - h(X[i, :, k], k)))
            end
        elseif Γ[k] == 0
            for i = 1:N
                # Using Constrained Bayesian Estimation
                D = MvNormal(h(X[i, :, k], k), sqrt.(cov(v)))
                yp = h(X[i, :, k], k) + rand(v)

                if sum(abs.(Z[:, k] - yp) .< δ) == 0
                    W[i, k] = log(Wr[i]) + log(pdf(D, yp))
                else
                    W[i, k] = -Inf
                end
            end
        end

        W_max = maximum(W[:, k])
        if W_max > -Inf
            W_tmp = exp.(W[:, k] - W_max*ones(N, 1))
            W[:, k] = W_tmp ./ sum(W_tmp)
        else
            println("Warning BPF: resampling X")
            #X[:, :, k] = rand(par.X0, N)
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

    Q = sqrt.(cov(w))
    R = sqrt.(cov(v))

    nx = sys.nd[1]
    ny = sys.nd[2]

    N = par.N

    X = zeros(par.N, nx, sys.T)
    W = zeros(par.N, sys.T)
	V = zeros(par.N, sys.T)

    Z = zeros(ny, T)
    Γ = zeros(T)

    X[:, :, 1] = rand(par.X0, N)
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

	Vn = eye(ny)
	for k = 1:ny
		Vn[k, k] *= (2*δ[k] / (M-1)) / sqrt(2)
	end
    yh = Array{Any}(M^ny)
    # ===

    JP_mu(x, k, H0, H1, H2) =
		[f(x, k), H0 + H2/2*diag(Q)]
	JP_sig(x, S, H0, H1, H2) =
		[[Q] [Q*H1'];
        [H1*Q] [H1*Q*H1' + 1/2*H2*Q*Q*H2' + S]]

    for k = 2:T

        # Run event kernel, SOD
        if sum(abs.(Z[:, k-1] - y[:, k]) .> δ) > 0
            Γ[k] = 1
            Z[:, k] = y[:, k]
        else
            Γ[k] = 0
            Z[:, k] = Z[:, k-1]

			#=
            # Works for 2D, make more general!
			if ny != 2
				error("Discretization not implemented for ny != 2")
			end

			ytmp = zeros(ny, M)
			for i = 1:2
				ytmp[i, :] = vcat(linspace(Z[i, k] - δ[i], Z[i, k] + δ[i], M)...)
			end
			tmp = reshape(collect(Iterators.product(ytmp[1, :], ytmp[2, :])), M^ny)
			yh = [[i[1], i[2]] for i in tmp]
			=#
			yh = vcat(linspace(Z[:, k] - δ, Z[:, k] + δ, M)...)

        end


        # Calculate auxiliary weights
        if Γ[k] == 1
            for i = 1:par.N

				H0 = h(f(X[i, :, k-1], k), k)
				H1, H2 = generate_H1H2(X[i, :, k-1], h, k)

                μ = JP_mu(X[i, :, k-1], k, H0, H1, H2)
                Σ = JP_sig(X[i, :, k-1], R, H0, H1, H2)

                q_aux_list[i] = MvNormal(μ[2], sqrt.(Σ[2,2]))
                V[i, k-1] = log(W[i, k-1]) + log(pdf(q_aux_list[i], y[:,k]))
        	end
        else
            for i = 1:par.N

				H0 = h(f(X[i, :, k-1], k), k)
				H1, H2 = generate_H1H2(X[i, :, k-1], h, k)

                μ = JP_mu(X[i, :, k-1], k, H0, H1, H2)
                Σ = JP_sig(X[i, :, k-1], R + Vn, H0, H1, H2)

                q_aux_list[i] = MvNormal(μ[2], sqrt.(Σ[2,2]))
                predLh = 0
                for j = 1:M^ny
                    predLh += pdf(q_aux_list[i], yh[j, :])
                end
				predLh /= M^ny

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

        #V[:, k-1] = W[:, k-1]

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
				H0 = h(f(Xr[i, :], k), k)
				H1, H2 = generate_H1H2(Xr[i, :], h, k)

                μ = JP_mu(Xr[i, :], k, H0, H1, H2)
                Σ = JP_sig(Xr[i, :], R, H0, H1, H2)
                S = fix_sym(Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2]')

                # 2*Σ ?
                q_list[i] = MvNormal(μ[1] + Σ[1,2]*inv(Σ[2,2])*(y[:,k] - μ[2]), sqrt.(S))

				#Ck = hd(Xr[i, :])
				#S = (Q\eye(4) + Ck'*(R\Ck))\eye(4)
				#m = S*(Q\f(Xr[i, :]) + Ck'*(R\y[:, k]))
				#q_list[i] = MvNormal(m, fix_sym(S))

				X[i, :, k] = rand(q_list[i])
            end
        else
            for i = 1:par.N
				H0 = h(f(Xr[i, :], k), k)
				H1, H2 = generate_H1H2(Xr[i, :], h, k)

                μ = JP_mu(Xr[i, :], k, H0, H1, H2)
                Σ = JP_sig(Xr[i, :], R + Vn, H0, H1, H2)

                μ_func(x) = μ[1] + Σ[1,2]*inv(Σ[2,2])*(x - μ[2])
                S = sqrt.(Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2]')

                wh = zeros(M^ny)
                for j = 1:M
                    wh[j] = log(pdf(MvNormal(μ[2], sqrt.(Σ[2, 2])), yh[j, :]))
                end

                wh_max = maximum(wh)
                if wh_max > -Inf
                    wh_tmp = exp.(wh - wh_max*ones(M^ny))
                    wh = wh_tmp ./ sum(wh_tmp)
                else
                    println("Bad conditioned weights for Mixture Gaussian; resetting to uniform")
                    wh = 1 / (M^ny) * ones(M^ny)
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
                    W[i, k] = log(Wr[i]) + log(pdf(v, Z[:,k] - h(X[i, :, k], k))) +
                                log(pdf(w, X[i, :, k] - f(Xr[i, :], k))) -
                                log(pdf(q_list[i], X[i, :, k])) -
                                log(pdf(q_aux_list[i], Z[:, k]))
                else
                    W[i, k] = log(Wr[i]) + log(pdf(v, Z[:,k] - h(X[i, :, k], k))) +
                                log(pdf(w, X[i, :, k] - f(Xr[i, :], k))) -
                                log(pdf(q_list[i], X[i, :, k]))
                end
            end
        else
            for i = 1:par.N
                # Likelihood
				D = Normal(h(X[i, :, k], k)[1], R[1])

				#py = cdf(D, Z[k]) - cdf(D, Z[k])
				yp = h(X[i, :, k], k) + rand(v)

                if sum(abs.(Z[:, k] - yp) .< δ) == 0
                    py = pdf(D, yp[1])
                else
                    py = 0
                end

                # Propagation

                px = pdf(w, X[i, :, k] - f(Xr[i, :], k))

                # proposal distribution
                q = pdf(q_list[i], X[i, :, k])

                if res[k] == 1
                    # Predictive likelihood
                    pL = 0
                    for j = 2:M
                        pL += pdf(q_aux_list[i], yh[j, :])
                    end
					pL /= M^ny

                    W[i, k] = log(Wr[i]) + log(py) + log(px) - log(q) - log(pL)
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
            println("Warning APF: resampling X")
            #X[:, :, 1] = rand(par.X0, N)
            W[:, k] = 1/par.N * ones(par.N, 1)
            fail[k] = 1
            Neff[k] = 0
        end


    end

    return X, W, Z, Γ, Neff, res, fail
end

"""
Event driven particle filters
"""
function ebpf_mbt(y, sys, par, δ)
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


	X[:, :, 1] = rand(par.X0, N)
    W[:, 1] = 1/N .* ones(N, 1)

    Neff = zeros(T)
    res = zeros(T)
    fail = zeros(T)

    N_T = N / 2

    idx = collect(1:N)
    Xr = X[:, 1]
    Wr = W[:, 1]

	xh = zeros(nx, T)
	xh[:, 1] = W[:, 1]' * X[:, :, 1]

    for k = 2:T

        # Run event kernel, SOD

		xh[:, k] = f(xh[:, k-1], k)
		y_pred = h(xh[:, k], k)
        if sum(abs.(y_pred - y[:, k]) .> δ) > 0
            Γ[k] = 1
            Z[:, k] = y[:, k]
        else
            Γ[k] = 0
            Z[:, k] = y_pred
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
                W[i, k] = log(Wr[i]) + log(pdf(sys.v, y[:, k] - h(X[i, :, k], k)))
            end
        elseif Γ[k] == 0
            for i = 1:N
                # Using Constrained Bayesian Estimation
                D = MvNormal(h(X[i, :, k], k), sqrt.(cov(v)))
                yp = h(X[i, :, k], k) + rand(v)

                if sum(abs.(Z[:, k] - yp) .< δ) == 0
                    W[i, k] = log(Wr[i]) + log(pdf(D, yp))
                else
                    W[i, k] = -Inf
                end
            end
        end

        W_max = maximum(W[:, k])
        if W_max > -Inf
            W_tmp = exp.(W[:, k] - W_max*ones(N, 1))
            W[:, k] = W_tmp ./ sum(W_tmp)
        else
            println("Warning BPF: resampling X")
            #X[:, :, k] = rand(par.X0, N)
            W[:, k] = 1/N .* ones(N, 1)
            fail[k] = 1
            Neff[k] = 0
        end

		if Γ[k] == 1
			xh[:, k] = W[:, k]'*X[:, :, k]
		end

    end
    return X, W, Z, Γ, Neff, res, fail
end

function eapf_mbt(y, sys, par, δ)
    """
    Event-based auxiliary particle filter
    """
    f = sys.f
    h = sys.h
    w = sys.w
    v = sys.v

    Q = sqrt.(cov(w))
    R = sqrt.(cov(v))

    nx = sys.nd[1]
    ny = sys.nd[2]

    N = par.N

    X = zeros(par.N, nx, sys.T)
    W = zeros(par.N, sys.T)
	V = zeros(par.N, sys.T)

    Z = zeros(ny, T)
    Γ = zeros(T)

    X[:, :, 1] = rand(par.X0, N)
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

	Vn = eye(ny)
	for k = 1:ny
		Vn[k, k] *= (2*δ[k] / (M-1)) / sqrt(2)
	end
    yh = Array{Any}(M^ny)
    # ===

    JP_mu(x, k, H0, H1, H2) =
		[f(x, k), H0 + H2/2*diag(Q)]
	JP_sig(x, S, H0, H1, H2) =
		[[Q] [Q*H1'];
        [H1*Q] [H1*Q*H1' + 1/2*H2*Q*Q*H2' + S]]

	xh = zeros(nx, T)
	xh[:, 1] = W[:, 1]' * X[:, :, 1]

    for k = 2:T

		# MBT
		xh[:, k] = f(xh[:, k-1], k)
		y_pred = h(xh[:, k], k)
		if sum(abs.(y_pred - y[:, k]) .> δ) > 0
			Γ[k] = 1
			Z[:, k] = y[:, k]
		else
			Γ[k] = 0
			Z[:, k] = y_pred
			yh = vcat(linspace(Z[:, k] - δ, Z[:, k] + δ, M)...)

		end

        # Calculate auxiliary weights
        if Γ[k] == 1
            for i = 1:par.N

				H0 = h(f(X[i, :, k-1], k), k)
				H1, H2 = generate_H1H2(X[i, :, k-1], h, k)

                μ = JP_mu(X[i, :, k-1], k, H0, H1, H2)
                Σ = JP_sig(X[i, :, k-1], R, H0, H1, H2)

                q_aux_list[i] = MvNormal(μ[2], sqrt.(Σ[2,2]))
                V[i, k-1] = log(W[i, k-1]) + log(pdf(q_aux_list[i], y[:,k]))
        	end
        else
            for i = 1:par.N

				H0 = h(f(X[i, :, k-1], k), k)
				H1, H2 = generate_H1H2(X[i, :, k-1], h, k)

                μ = JP_mu(X[i, :, k-1], k, H0, H1, H2)
                Σ = JP_sig(X[i, :, k-1], R + Vn, H0, H1, H2)

                q_aux_list[i] = MvNormal(μ[2], sqrt.(Σ[2,2]))
                predLh = 0
                for j = 1:M^ny
                    predLh += pdf(q_aux_list[i], yh[j, :])
                end
				predLh /= M^ny

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

        #V[:, k-1] = W[:, k-1]

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
				H0 = h(f(Xr[i, :], k), k)
				H1, H2 = generate_H1H2(Xr[i, :], h, k)

                μ = JP_mu(Xr[i, :], k, H0, H1, H2)
                Σ = JP_sig(Xr[i, :], R, H0, H1, H2)
                S = fix_sym(Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2]')

                # 2*Σ ?
                q_list[i] = MvNormal(μ[1] + Σ[1,2]*inv(Σ[2,2])*(y[:,k] - μ[2]), sqrt.(S))

				#Ck = hd(Xr[i, :])
				#S = (Q\eye(4) + Ck'*(R\Ck))\eye(4)
				#m = S*(Q\f(Xr[i, :]) + Ck'*(R\y[:, k]))
				#q_list[i] = MvNormal(m, fix_sym(S))

				X[i, :, k] = rand(q_list[i])
            end
        else
            for i = 1:par.N
				H0 = h(f(Xr[i, :], k), k)
				H1, H2 = generate_H1H2(Xr[i, :], h, k)

                μ = JP_mu(Xr[i, :], k, H0, H1, H2)
                Σ = JP_sig(Xr[i, :], R + Vn, H0, H1, H2)

                μ_func(x) = μ[1] + Σ[1,2]*inv(Σ[2,2])*(x - μ[2])
                S = sqrt.(Σ[1,1] - Σ[1,2]*inv(Σ[2,2])*Σ[1,2]')

                wh = zeros(M^ny)
                for j = 1:M
                    wh[j] = log(pdf(MvNormal(μ[2], sqrt.(Σ[2, 2])), yh[j, :]))
                end

                wh_max = maximum(wh)
                if wh_max > -Inf
                    wh_tmp = exp.(wh - wh_max*ones(M^ny))
                    wh = wh_tmp ./ sum(wh_tmp)
                else
                    println("Bad conditioned weights for Mixture Gaussian; resetting to uniform")
                    wh = 1 / (M^ny) * ones(M^ny)
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
                    W[i, k] = log(Wr[i]) + log(pdf(v, Z[:,k] - h(X[i, :, k], k))) +
                                log(pdf(w, X[i, :, k] - f(Xr[i, :], k))) -
                                log(pdf(q_list[i], X[i, :, k])) -
                                log(pdf(q_aux_list[i], Z[:, k]))
                else
                    W[i, k] = log(Wr[i]) + log(pdf(v, Z[:,k] - h(X[i, :, k], k))) +
                                log(pdf(w, X[i, :, k] - f(Xr[i, :], k))) -
                                log(pdf(q_list[i], X[i, :, k]))
                end
            end
        else
            for i = 1:par.N
                # Likelihood
				D = Normal(h(X[i, :, k], k)[1], R[1])

				#py = cdf(D, Z[k]) - cdf(D, Z[k])
				yp = h(X[i, :, k], k) + rand(v)

                if sum(abs.(Z[:, k] - yp) .< δ) == 0
                    py = pdf(D, yp[1])
                else
                    py = 0
                end

                # Propagation

                px = pdf(w, X[i, :, k] - f(Xr[i, :], k))

                # proposal distribution
                q = pdf(q_list[i], X[i, :, k])

                if res[k] == 1
                    # Predictive likelihood
                    pL = 0
                    for j = 2:M
                        pL += pdf(q_aux_list[i], yh[j, :])
                    end
					pL /= M^ny

                    W[i, k] = log(Wr[i]) + log(py) + log(px) - log(q) - log(pL)
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
            println("Warning APF: resampling X")
            #X[:, :, 1] = rand(par.X0, N)
            W[:, k] = 1/par.N * ones(par.N, 1)
            fail[k] = 1
            Neff[k] = 0
        end

		if Γ[k] == 1
			xh[:, k] = W[:, k]'*X[:, :, k]
		end

    end

    return X, W, Z, Γ, Neff, res, fail
end

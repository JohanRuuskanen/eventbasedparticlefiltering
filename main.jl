using PyPlot
using Distributions

function sim_sys(f, h, T, nd)
    x = zeros(nd[1], T)
    y = zeros(nd[2], T)

    for k = 2:T
        x[:, k] = rand(f(x[:, k-1], k))
        y[:, k] = rand(h(x[:, k], k))
    end

    return x, y
end

function MBT_event(y, z, xh, f, h, δ, k)

    xh = mean(f(xh, k))
    yh = mean(h(xh, k))

    γ = 0
    if abs(yh - y)[1] > δ
        γ = 1
        z = y
    end

    return z, γ, xh, yh

end

function ebpf(y, q, h, T, N, nd)

    X = zeros(N, nd[1], T)
    W = zeros(N, T)

    X[:, :, 1] = rand(q(zeros(nd[1]), 0), N)
    W[:, 1] = 1/N .* ones(N, 1)

    idx = collect(1:N)
    for k = 2:T

        # Propagate
        for i = 1:N
            X[i, :, k] = rand(q(X[idx[i], :, k-1], k))
        end

        # Weight
        for i = 1:N
            W[i, k] = pdf(h(X[i, :, k], k), y[:, k])
        end
        W[:, k] = W[:, k] ./ sum(W[:, k])

        # Resample using systematic resampling
        #idx = rand(Categorical(W[:, k]), N)
        wc = cumsum(W[:, k])
        u = (([0:(N-1)] + rand()) / N)[1]
        c = 1
        for i = 1:N
            while wc[c] < u[i]
                c = c + 1
            end
            idx[i] = c
        end

    end

    return X, W

end

# Parameters
N = 1000
T = 100
δ = 5

# Nonlinear and non-Gaussian system
f(x, t) = MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos(1.2*t), 10*eye(1))
h(x, t) = MvNormal(x.^2/20, 1*eye(1))

nd = [1, 1]
x, y = sim_sys(f, h, T, nd)

"""
# For estimation
q(x, t) = MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos(1.2*t), 10*eye(1))
h(x, t) = MvNormal(x.^2/20, 1*eye(1))

X, W = ebpf(y, q, h, T, N, nd)

x_h = zeros(nd[1], T)
for k = 1:nd[1]
    x_h[k, :] = mean(diag(W'*X[:, k, :]), 2)
end
"""
# Test MBT event kernel
Z = zeros(1, T)
Γ = zeros(1, T)
Xh = zeros(1, T)
Yh = zeros(1, T)
for k = 2:T
    Z[:, k], Γ[:, k], Xh[:, k], Yh[:, k] = MBT_event(y[:, k], Z[:, k-1], Xh[:, k-1], f, h, δ, k)
end
#(y, z, xh, f, h, δ, k)
figure(1)
clf()
subplot(nd[1]+1, 1, 1)
plot(1:T, y')
plt[:step](1:T, Z', "g")
plot(1:T, Yh', "g--")
title("Measurements")
for k = 2:(nd[1]+1)
    subplot(nd[1]+1, 1, k)
    plot(1:T, x[k-1, :])
    #plot(1:T, x_h[k-1, :], "r--")
    plot(1:T, Xh[:], "g--")
    title("State $k")
end

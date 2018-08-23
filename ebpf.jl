using PyPlot
using Distributions

function sim_system(F, B, H, D, u, T, σ_e, σ_v)
    n1 = size(F, 1)
    n2 = size(H, 1)
    
    x = zeros(n1, T)
    y = zeros(n2, T)

    for k = 2:T
        x[:, k] = F*x[:, k-1] + B*u[k] + σ_e*randn(n1, 1)
        y[k] = (H*x[:, k])[1] + σ_v*randn()
    end

    return x, y
end

function bpf(y, F, H, T, Q, R) 
    n1 = size(F, 1)
    n2 = size(H, 1)

    X = zeros(N, 3, T)
    W = zeros(N, T)

    X[:, :, 1] = rand(MvNormal(zeros(n1), Q), N)' 
    W[:, 1] = 1/N * ones(N, 1)

    idx = 1:N

    for k = 2:T
        # Propagation
        x_bar = X[idx, :, k-1]
        for i = 1:N
            X[i, :, k] = rand(MvNormal(F*x_bar[i, :], Q))
        end

        # Weighting
        w = zeros(N, 1)
        for i = 1:N
            w[i] = pdf(Normal((H*X[i, :, k])[1], R), y[k])  
        end
        W[:, k] = w ./ sum(w)

        # Resampling
        idx = rand(Categorical(W[:, k]), N)

    end

    return X, W

end

N = 100000
T = 100
h = 1

A = [0 1; 0 0]
B = [0; 1]
C = [1 0]
D = 0

σ_e = 0.01
σ_v = 10


u = zeros(1, T)
u[10:30] = 1
u[40:60] = -2
u[70:80] = 1

F = expm([A B; zeros(1, size(A, 2)) zeros(1, 1)])
H = hcat(C, 0)
Φ = F[1:size(A, 1), 1:size(A, 2)]
Γ = F[1:size(B, 1), end]

x, y = sim_system(Φ, Γ, C, D, u, T, σ_e, σ_v);#σ_e, σ_v)
X, W = bpf(y, F, H, T, [1 0 0; 0 1 0; 0 0 0.5],σ_v)#σ_e*eye(3), σ_v)  

x_h = [diag(W'*X[:, 1, :]) diag(W'*X[:, 2, :]) diag(W'*X[:, 3, :])]


figure(1)
clf()
subplot(4, 1, 1)
plot(y')
subplot(4, 1, 2)
plot(x[1, :])
plot(x_h[:, 1], "r--")
subplot(4, 1, 3)
plot(x[2, :])
plot(x_h[:, 2], "r--")
subplot(4, 1, 4)
plot(u')
plot(x_h[:, 3], "r--")


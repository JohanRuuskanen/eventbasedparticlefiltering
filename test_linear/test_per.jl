using JLD
using PyPlot
using StatsBase
using Distributions

include("../src/misc.jl")
include("../src/event_kernels.jl")

include("filters.jl")

# Parameters
N = 100
T = 200
δ = 5

# Nonlinear and non-Gaussian system
#f(x, t) = MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos(1.2*t), 10*eye(1))
#h(x, t) = MvNormal(x.^2/20, 0.1*eye(1)) #MvNormal(atan.(x), 1*eye(1))
#nd = [1, 1]

A = [0.8 1; 0 0.95]
C = [0.7 0.6]

Q = 0.1*eye(2)
R = 0.01*eye(1)

sys = lin_sys_params(A, C, Q, R, T)
x, y = sim_lin_sys(sys)

nx = size(A, 1)
ny = size(C, 1)

# For estimation
par = pf_params(N)

X_bpf, W_bpf = bpf(y, sys, par)
X_apf, W_apf = apf(y, sys, par)

xh_bpf = zeros(nx, T)
xh_apf = zeros(nx, T)
for k = 1:nx
    xh_bpf[k, :] = sum(diag(W_bpf'*X_bpf[:, k, :]), 2)
    xh_apf[k, :] = sum(diag(W_apf'*X_apf[:, k, :]), 2)
end

err_bpf = x - xh_bpf
err_apf = x - xh_apf

println("")
println("Total error")
println("BPF: $(mean(err_bpf.^2))")
println("APF: $(mean(err_apf.^2))")

figure(1)
clf()
subplot(3, 1, 1)
plot(y')
subplot(3, 1, 2)
plot(x[1, :])
plot(xh_bpf[1, :])
plot(xh_apf[1, :])
legend(["True", "BPF", "APF"])
subplot(3, 1, 3)
plot(x[2, :])
plot(xh_bpf[2, :])
plot(xh_apf[2, :])
legend(["True", "BPF", "APF"])

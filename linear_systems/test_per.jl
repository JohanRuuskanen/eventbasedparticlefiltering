using JLD
using PyPlot
using StatsBase
using Distributions
using LinearAlgebra

include("../src/misc.jl")
include("../src/event_kernels.jl")

include("filters.jl")

# Parameters
N = 100
T = 200

A = [0.8 1; 0 0.95]
C = [0.7 0.6]

Q = 1*Matrix{Float64}(I,2,2)
R = 0.1*Matrix{Float64}(I,1,1)

sys = lin_sys_params(A, C, Q, R, T)
x, y = sim_lin_sys(sys)

nx = size(A, 1)
ny = size(C, 1)

# For estimation
par = pf_params(N)

X_bpf, W_bpf, N_eff_bpf = bpf(y, sys, par)
X_apf, W_apf, N_eff_apf = apf(y, sys, par)

xh_bpf = zeros(nx, T)
xh_apf = zeros(nx, T)
for k = 1:nx
    xh_bpf[k, :] = sum(diag(W_bpf'*X_bpf[:, k, :]), dims=2)
    xh_apf[k, :] = sum(diag(W_apf'*X_apf[:, k, :]), dims=2)
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

figure(2)
clf()
subplot(2, 1, 1)
title("Plot N_eff BPF")
plot(1:T, N_eff_bpf)
ylim([0, 1.2*N])
subplot(2, 1, 2)
title("Plot N_eff APF")
plot(1:T, N_eff_apf)
ylim([0, 1.2*N])

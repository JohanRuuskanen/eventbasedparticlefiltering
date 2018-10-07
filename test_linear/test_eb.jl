using JLD
using PyPlot
using StatsBase
using Distributions

include("../src/misc.jl")
#include("../src/event_kernels.jl")

#include("filters.jl")
include("filters_eventbased.jl")

# Parameters
N = 100
T = 200
δ = 4

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

X_ebpf, W_ebpf, Z_ebpf, Γ_ebpf, Neff_ebpf, res_ebpf, fail_ebpf = ebpf(y, sys, par, δ)
X_eapf, W_eapf, Z_eapf, Γ_eapf, Neff_eapf, res_eapf, fail_eapf = eapf(y, sys, par, δ)

xh_ebpf = zeros(nx, T)
xh_eapf = zeros(nx, T)
for k = 1:nx
    xh_ebpf[k, :] = sum(diag(W_ebpf'*X_ebpf[:, k, :]), 2)
    xh_eapf[k, :] = sum(diag(W_eapf'*X_eapf[:, k, :]), 2)
end

err_ebpf = x - xh_ebpf
err_eapf = x - xh_eapf

idx_bpf = find(x -> x == 1, Γ_ebpf)
idx_apf = find(x -> x == 1, Γ_eapf)

println("")
println("Total error")
println("EBPF x1: $(mean(err_ebpf[1, :].^2))")
println("EBPF x2: $(mean(err_ebpf[2, :].^2))")
println("")
println("EAPF x1: $(mean(err_eapf[1, :].^2))")
println("EAPF x2: $(mean(err_eapf[2, :].^2))")
println("")
println("Total error in new points")
println("EBPF x1: $(mean(err_ebpf[1, idx_bpf].^2))")
println("EBPF x2: $(mean(err_ebpf[2, idx_apf].^2))")
println("")
println("EAPF x1: $(mean(err_eapf[1, idx_bpf].^2))")
println("EAPF x2: $(mean(err_eapf[2, idx_apf].^2))")
println("")
println("Special")
println("EBPF Neff: $(mean(Neff_ebpf))")
println("EBPF res: $(sum(res_ebpf))")
println("EBPF fail: $(sum(fail_ebpf))")
println("")
println("EAPF Neff: $(mean(Neff_eapf))")
println("EAPF res: $(sum(res_eapf))")
println("EAPF fail: $(sum(fail_eapf))")
println("")

figure(1)
clf()
subplot(3, 1, 1)
plot(y')
plot(Z_ebpf')
plot(Z_eapf')
legend(["y", "z BPF", "z APF"])
subplot(3, 1, 2)
plot(x[1, :])
plot(xh_ebpf[1, :])
plot(xh_eapf[1, :])
legend(["True", "EBPF", "EAPF"])
subplot(3, 1, 3)
plot(x[2, :])
plot(xh_ebpf[2, :])
plot(xh_eapf[2, :])
legend(["True", "EBPF", "EAPF"])


idx_res_ebpf = find(x->x == 1, res_ebpf)
idx_fail_ebpf = find(x->x == 1, fail_ebpf)
idx_res_eapf = find(x->x == 1, res_eapf)
idx_fail_eapf = find(x->x == 1, fail_eapf)
figure(2)
clf()
title("Effective sample size")
subplot(2, 1, 1)
plot(1:T, Neff_ebpf, "C0")
plot((1:T)[idx_res_ebpf], Neff_ebpf[idx_res_ebpf], "C0o")
plot((1:T)[idx_fail_ebpf], Neff_ebpf[idx_fail_ebpf], "C0x")
subplot(2, 1, 2)
plot(1:T, Neff_eapf, "C0")
plot((1:T)[idx_res_eapf], Neff_eapf[idx_res_eapf], "C0o")
plot((1:T)[idx_fail_eapf], Neff_eapf[idx_fail_eapf], "C0x")


k = 150 #154
dx = linspace(-10, 10, 1000)
bins = -10:0.25:10
figure(3)
clf()
subplot(4, 1, 1)
plt[:hist](X_ebpf[:, 1, k],bins=bins, weights=W_ebpf[:, k], density=true, alpha=0.5)
plt[:hist](X_ebpf[:, 1, k],bins=bins, density=true, alpha=0.5)
xlim([-10, 10])
subplot(4, 1, 2)
plt[:hist](X_eapf[:, 1, k],bins=bins, weights=W_eapf[:, k], density=true, alpha=0.5)
plt[:hist](X_eapf[:, 1, k],bins=bins, density=true, alpha=0.5)
xlim([-10, 10])
subplot(4, 1, 3)
plot(1:N, sort(W_ebpf[:, k]))
plot(1:N, sort(W_eapf[:, k]))
subplot(4, 1, 4)
idx_bpf = sortperm(X_ebpf[:, 1, k])
idx_apf = sortperm(X_eapf[:, 1, k])
plot(X_ebpf[idx_bpf, 1, k], W_ebpf[idx_bpf, k], 'o')
plot(X_eapf[idx_apf, 1, k], W_eapf[idx_apf, k], 'x')
xlim([-10, 10])

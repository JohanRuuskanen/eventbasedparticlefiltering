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
T = 1000
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

X_ebpf, W_ebpf, Z_ebpf, Γ_ebpf, Neff_ebpf, fail_ebpf = ebpf(y, sys, par, δ)
X_esis, W_esis, Z_esis, Γ_esis, Neff_esis, fail_esis = esis(y, sys, par, δ)
X_eapf, W_eapf, Z_eapf, Γ_eapf, Neff_eapf, fail_eapf = eapf(y, sys, par, δ)

xh_ebpf = zeros(nx, T)
xh_esis = zeros(nx, T)
xh_eapf = zeros(nx, T)
for k = 1:nx
    xh_ebpf[k, :] = sum(diag(W_ebpf'*X_ebpf[:, k, :]), 2)
    xh_esis[k, :] = sum(diag(W_esis'*X_esis[:, k, :]), 2)
    xh_eapf[k, :] = sum(diag(W_eapf'*X_eapf[:, k, :]), 2)
end

err_ebpf = x - xh_ebpf
err_esis = x - xh_esis
err_eapf = x - xh_eapf

println("")
println("Total error")
println("EBPF x1: $(mean(err_ebpf[1, :].^2))")
println("EBPF x2: $(mean(err_ebpf[2, :].^2))")
println("")
println("ESIS x1: $(mean(err_esis[1, :].^2))")
println("ESIS x2: $(mean(err_esis[2, :].^2))")
println("")
println("EAPF x1: $(mean(err_eapf[1, :].^2))")
println("EAPF x2: $(mean(err_eapf[2, :].^2))")


figure(1)
clf()
subplot(3, 1, 1)
plot(y')
plot(Z_ebpf')
plot(Z_esis')
plot(Z_eapf')
legend(["y", "z BPF", "z SIS", "z APF"])
subplot(3, 1, 2)
plot(x[1, :])
plot(xh_ebpf[1, :])
plot(xh_esis[1, :])
plot(xh_eapf[1, :])
legend(["True", "EBPF", "ESIS", "EAPF"])
subplot(3, 1, 3)
plot(x[2, :])
plot(xh_ebpf[2, :])
plot(xh_esis[2, :])
plot(xh_eapf[2, :])
legend(["True", "EBPF", "ESIS", "EAPF"])

idx_ebpf = find(x -> x == 1, Γ_ebpf)
idx_esis = find(x -> x == 1, Γ_esis)
idx_eapf = find(x -> x == 1, Γ_eapf)
figure(2)
clf()
subplot(3, 1, 1)
plot(1:T, Neff_ebpf)
plot(1:T, Neff_esis)
plot(1:T, Neff_eapf)
subplot(3, 1, 2)
plot(idx_ebpf, Neff_ebpf[idx_ebpf], "x-")
plot(idx_esis, Neff_esis[idx_esis], "x-")
plot(idx_eapf, Neff_eapf[idx_eapf], "x-")
subplot(3, 1, 3)
plot(1:T, fail_ebpf, "o")
plot(1:T, fail_esis, "o")
plot(1:T, fail_eapf, "o")

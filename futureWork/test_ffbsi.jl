
using PyPlot
using Distributions

include("../src/misc.jl")
include("filters.jl")
include("filters_eventbased.jl")

# Parameters
N = 200
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
M = 50

X_ebpf, W_ebpf, Z_ebpf, Γ_ebpf, Neff_ebpf, fail_ebpf = ebpf(y, sys, par, δ)
X_eapf, W_eapf, Z_eapf, Γ_eapf, Neff_eapf, fail_eapf = eapf(y, sys, par, δ)

Xs_ebpf = FFBSi(X_ebpf, W_ebpf, sys, M)
Xs_eapf = FFBSi(X_eapf, W_eapf, sys, M)

xh_ebpf = zeros(nx, T)
xh_eapf = zeros(nx, T)
xh_psa = zeros(nx, T)
xh_psb = zeros(nx, T)
for k = 1:nx
    xh_ebpf[k, :] = sum(diag(W_ebpf'*X_ebpf[:, k, :]), 2)
    xh_eapf[k, :] = sum(diag(W_eapf'*X_eapf[:, k, :]), 2)
    xh_psa[k, :] = sum(1/M*Xs_ebpf[:, k, :], 1)
    xh_psb[k, :] = sum(1/M*Xs_eapf[:, k, :], 1)
end

err_ebpf = x - xh_ebpf
err_eapf = x - xh_eapf
err_psa = x - xh_psa
err_psb = x - xh_psb

println("")
println("Total error")
println("EBPF x1: $(mean(err_ebpf[1, :].^2))")
println("EBPF x2: $(mean(err_ebpf[2, :].^2))")
println("")
println("EAPF x1: $(mean(err_eapf[1, :].^2))")
println("EAPF x2: $(mean(err_eapf[2, :].^2))")
println("")
println("PSA x1: $(mean(err_psa[1, :].^2))")
println("PSA x2: $(mean(err_psa[2, :].^2))")
println("")
println("PSB x1: $(mean(err_psb[1, :].^2))")
println("PSB x2: $(mean(err_psb[2, :].^2))")


figure(1)
clf()
subplot(2, 1, 1)
plot(1:T, x[1, :])
plot(1:T, xh_eapf[1, :])
plot(1:T, xh_psb[1, :])
subplot(2, 1, 2)
plot(1:T, x[2, :])
plot(1:T, xh_eapf[2, :])
plot(1:T, xh_psb[2, :])

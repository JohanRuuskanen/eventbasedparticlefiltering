using JLD
using PyPlot
using StatsBase
using Distributions

include("../src/misc.jl")
include("filters.jl")
include("filters_eventbased.jl")

# Parameters
N = 50
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

X_ebpf, W_ebpf, xh_ebpf, yh_ebpf, Z_ebpf, Γ_ebpf = ebpf(y, sys, par, δ)
X_esis, W_esis, xh_esis, yh_esis, Z_esis, Γ_esis = esis(y, sys, par, δ)
X_eapf, W_eapf, xh_eapf, yh_eapf, Z_eapf, Γ_eapf = eapf(y, sys, par, δ)

xh_ebse, P_ebse, Z_ebse, Γ_ebse = ebse(y, sys, δ)
xh_kal, P_kal = kalman_filter(y, sys)

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
err_ebse = x - xh_ebse
err_kal = x - xh_kal

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
println("")
println("EBSE x1: $(mean(err_ebse[1, :].^2))")
println("EBSE x2: $(mean(err_ebse[2, :].^2))")
println("")
println("Kal x1: $(mean(err_kal[1, :].^2))")
println("Kal x2: $(mean(err_kal[2, :].^2))")


figure(1)
clf()
subplot(3, 1, 1)
plot(y')
plot(Z_ebpf')
plot(Z_esis')
plot(Z_eapf')
plot(Z_ebse')
legend(["y", "z BPF", "z SIS", "z APF", "z EBSE"])
subplot(3, 1, 2)
plot(x[1, :])
plot(xh_ebpf[1, :])
plot(xh_esis[1, :])
plot(xh_eapf[1, :])
plot(xh_ebse[1, :])
plot(xh_kal[1, :])
legend(["True", "EBPF", "ESIS", "EAPF", "EBSE", "kal"])
subplot(3, 1, 3)
plot(x[2, :])
plot(xh_ebpf[2, :])
plot(xh_esis[2, :])
plot(xh_eapf[2, :])
plot(xh_ebse[2, :])
plot(xh_kal[2, :])
legend(["True", "EBPF", "ESIS", "EAPF", "EBSE", "kal"])

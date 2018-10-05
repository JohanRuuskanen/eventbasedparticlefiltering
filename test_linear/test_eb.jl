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

X_ebpf, W_ebpf, Z_ebpf, Γ_ebpf, Neff_ebpf, fail_ebpf, res_ebpf = ebpf(y, sys, par, δ)
X_eapf, W_eapf, Z_eapf, Γ_eapf, Neff_eapf, fail_eapf, res_eapf = eapf(y, sys, par, δ)
#xh_ebse, ~, Z_ebse, Γ_ebse = ebse(y, sys, δ)

xh_ebpf = zeros(nx, T)
xh_eapf = zeros(nx, T)
for k = 1:nx
    xh_ebpf[k, :] = sum(diag(W_ebpf'*X_ebpf[:, k, :]), 2)
    xh_eapf[k, :] = sum(diag(W_eapf'*X_eapf[:, k, :]), 2)
end

err_ebpf = x - xh_ebpf
err_eapf = x - xh_eapf
#err_ebse = x - xh_ebse

idx_bpf = find(x -> x == 1, Γ_ebpf)
idx_apf = find(x -> x == 1, Γ_eapf)
#idx_ebse = find(x -> x == 1, Γ_ebse)

err_ebpf_t = x[:, idx_bpf] - xh_ebpf[:, idx_bpf]
err_eapf_t = x[:, idx_apf] - xh_eapf[:, idx_apf]
#err_ebse_t = x[:, idx_ebse] - xh_ebse[:, idx_ebse]

println("")
println("Total error")
println("EBPF x1: $(mean(err_ebpf[1, :].^2))")
println("EBPF x2: $(mean(err_ebpf[2, :].^2))")
println("")
println("EAPF x1: $(mean(err_eapf[1, :].^2))")
println("EAPF x2: $(mean(err_eapf[2, :].^2))")
println("")
println("Resamples")
println("EBPF: $(sum(res_ebpf))")
println("EAPF: $(sum(res_eapf))")
println("")
println("Failures")
println("EBPF: $(sum(fail_ebpf))")
println("EAPF: $(sum(fail_eapf))")
println("")
#println("EBSE x1: $(mean(err_ebse[1, :].^2)) ± $(std(err_ebse[1, :].^2))")
#println("EBSE x2: $(mean(err_ebse[2, :].^2)) ± $(std(err_ebse[2, :].^2))")

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

idx_ebpf = find(x->x==1, Γ_ebpf)
idx_res_ebpf = find(x->x==1, res_ebpf) - 1
idx_eapf = find(x->x==1, Γ_eapf)
idx_res_eapf = find(x->x==1, res_eapf) - 1
figure(2)
clf()
subplot(2, 1, 1)
plot(1:T, Neff_ebpf)
plot((1:T)[idx_res_ebpf], Neff_ebpf[idx_res_ebpf], "bx")
plot((1:T)[idx_ebpf], Neff_ebpf[idx_ebpf], "bo")
subplot(2, 1, 2)
plot(1:T, Neff_eapf)
plot((1:T)[idx_res_eapf], Neff_eapf[idx_res_eapf], "bx")
plot((1:T)[idx_eapf], Neff_eapf[idx_eapf], "bo")
#plot(1:T, Neff_eapf)

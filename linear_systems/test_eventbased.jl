using JLD
using PyPlot
using Random
using StatsBase
using Distributions
using LinearAlgebra

include("../funcs/misc.jl")
include("../funcs/plotting.jl")
include("filters_eventbased.jl")

Random.seed!(1)

# Parameters
N = 10
T = 20
δ = 2

A = [0.8 1; 0 0.95]
C = [0.7 0.6]

Q = 1*Matrix{Float64}(I, 2, 2)
R = 1*Matrix{Float64}(I, 1, 1)

sys = lin_sys_params(A, C, Q, R, T)
x, y = sim_lin_sys(sys)

nx = size(A, 1)
ny = size(C, 1)

# For estimation
par_bpf = pf_params(10)
par_apf = pf_params(500)

# Using benchmarktools or time?
# Create struct to incorporate output parameters?
X_ebpf, W_ebpf, Z_ebpf, Γ_ebpf, Neff_ebpf, res_ebpf, fail_ebpf, S = ebpf(y, sys, par_bpf, δ)
plot_particle_trace(X_ebpf[:,1,:], S, x_true=x[1,:])


X_eapf, W_eapf, Z_eapf, Γ_eapf, Neff_eapf, res_eapf, fail_eapf = eapf(y, sys, par_apf, δ)
xh_ebse, Ph_ebse, Z_ebse, Γ_ebse = ebse(y, sys, δ)

xh_ebpf = zeros(nx, T)
xh_eapf = zeros(nx, T)
for k = 1:nx
    xh_ebpf[k, :] = sum(diag(W_ebpf'*X_ebpf[:, k, :]), dims=2)
    xh_eapf[k, :] = sum(diag(W_eapf'*X_eapf[:, k, :]), dims=2)
end

err_ebpf = x - xh_ebpf
err_eapf = x - xh_eapf
err_ebse = x - xh_ebse

idx_bpf = findall(x -> x == 1, Γ_ebpf)
idx_apf = findall(x -> x == 1, Γ_eapf)
idx_ebse = findall(x -> x == 1, Γ_ebse)

err_ebpf_t = err_ebpf[:, idx_bpf]
err_eapf_t = err_eapf[:, idx_apf]
err_ebse_t = err_ebse[:, idx_ebse]

println("")
println("Time:")
println("EBPF t: $(a)")
println("EAPF t: $(b)")
println("EBSE t: $(c)")
println("")
println("Total error")
println("EBPF x1: $(mean(err_ebpf[1, :].^2))")
println("EBPF x2: $(mean(err_ebpf[2, :].^2))")
println("")
println("EAPF x1: $(mean(err_eapf[1, :].^2))")
println("EAPF x2: $(mean(err_eapf[2, :].^2))")
println("")
println("EBSE x1: $(mean(err_ebse[1, :].^2))")
println("EBSE x2: $(mean(err_ebse[2, :].^2))")
println("")
println("Error at new measurements")
println("EBPF x1: $(mean(err_ebpf_t[1, :].^2))")
println("EBPF x2: $(mean(err_ebpf_t[2, :].^2))")
println("")
println("EAPF x1: $(mean(err_eapf_t[1, :].^2))")
println("EAPF x2: $(mean(err_eapf_t[2, :].^2))")
println("")
println("EBSE x1: $(mean(err_ebse_t[1, :].^2))")
println("EBSE x2: $(mean(err_ebse_t[2, :].^2))")
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
plot(y[:])
plot(Z_ebpf[:])
plot(Z_eapf[:])
plot(Z_ebse[:])
legend(["y", "z BPF", "z APF", "z EBSE"])
subplot(3, 1, 2)
plot(x[1, :])
plot(xh_ebpf[1, :])
plot(xh_eapf[1, :])
plot(xh_ebse[1, :])
legend(["True", "EBPF", "EAPF", "EBSE"])
subplot(3, 1, 3)
plot(x[2, :])
plot(xh_ebpf[2, :])
plot(xh_eapf[2, :])
plot(xh_ebse[2, :])
legend(["True", "EBPF", "EAPF", "EBSE"])


idx_res_ebpf = findall(x->x == 1, res_ebpf)
idx_fail_ebpf = findall(x->x == 1, fail_ebpf)
idx_res_eapf = findall(x->x == 1, res_eapf)
idx_fail_eapf = findall(x->x == 1, fail_eapf)
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

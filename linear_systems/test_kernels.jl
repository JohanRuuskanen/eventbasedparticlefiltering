using JLD
using PyPlot
using Random
using StatsBase
using Distributions
using LinearAlgebra

include("../funcs/misc.jl")
include("../funcs/plotting.jl")
include("filters_eventbased.jl")

Random.seed!(2)

# Parameters
N = 100
T = 200
δ = 4

#A = [0.8 1; 0 0.95]
#C = [0.7 0.6]

A = reshape([0.9], (1, 1))
C = reshape([1.0], (1, 1))

nx = size(A, 1)
ny = size(C, 1)

Q = 1*Matrix{Float64}(I, nx, nx)
R = 0.1*Matrix{Float64}(I, ny, ny)

sys = lin_sys_params(A, C, Q, R, T)
x, y = sim_lin_sys(sys)

# For estimation
par1 = pf_params(N, "SOD", N)
par2 = pf_params(N, "MBT", N)

# Using benchmarktools or time?
# Create struct to incorporate output parameters?
X_bpf1, W_bpf1, Z_bpf1, Γ_bpf1, Neff_bpf1, res_bpf1, fail_bpf1, S_bpf1 = ebpf(y, sys, par1, δ)
X_bpf2, W_bpf2, Z_bpf2, Γ_bpf2, Neff_bpf2, res_bpf2, fail_bpf2, S_bpf2 = ebpf(y, sys, par2, δ)
X_apf1, W_apf1, Z_apf1, Γ_apf1, Neff_apf1, res_apf1, fail_apf1, S_apf1 = eapf(y, sys, par1, δ)
X_apf2, W_apf2, Z_apf2, Γ_apf2, Neff_apf2, res_apf2, fail_apf2, S_apf2 = eapf(y, sys, par2, δ)

figure(1)
clf()
subplot(2, 2, 1)
plot_data(y[:], Z_bpf1[:], δ=δ, nofig=true)
title("Using SOD BPF")
legend([L"y_k", L"z_k", L"z_k \pm   \delta"])
subplot(2, 2, 2)
plot_data(y[:], Z_bpf2[:], δ=δ, nofig=true)
title("Using MBT BPF")
legend([L"y_k", L"z_k", L"z_k \pm   \delta"])
subplot(2, 2, 3)
plot_data(y[:], Z_apf1[:], δ=δ, nofig=true)
title("Using SOD APF")
legend([L"y_k", L"z_k", L"z_k \pm   \delta"])
subplot(2, 2, 4)
plot_data(y[:], Z_apf2[:], δ=δ, nofig=true)
title("Using MBT APF")
legend([L"y_k", L"z_k", L"z_k \pm   \delta"])

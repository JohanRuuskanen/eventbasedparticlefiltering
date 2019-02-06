"""
Example for demonstrating the effects of resampling
"""

using JLD
using PyPlot
using Random
using StatsBase
using Distributions
using LinearAlgebra

include("../funcs/misc.jl")
include("../funcs/plotting.jl")
include("../linear_systems/filters_eventbased.jl")

Random.seed!(2)

# Parameters
N = 10
T = 200
δ = 4

A = reshape([1.0], (1, 1))
C = reshape([1.0], (1, 1))

nx = size(A, 1)
ny = size(C, 1)

Q = 1*Matrix{Float64}(I, nx, nx)
R = 0.1*Matrix{Float64}(I, ny, ny)

sys = lin_sys_params(A, C, Q, R, T)
x, y = sim_lin_sys(sys)

# For estimation
par_bpf = pf_params(N, "MBT", N)
par_apf = pf_params(N, "MBT", N)

# Using benchmarktools or time?
# Create struct to incorporate output parameters?
X_ebpf, W_ebpf, Z_ebpf, Γ_ebpf, Neff_ebpf, res_ebpf, fail_ebpf, S_ebpf = ebpf(y, sys, par_bpf, δ)
X_eapf, W_eapf, Z_eapf, Γ_eapf, Neff_eapf, res_eapf, fail_eapf, S_eapf = eapf(y, sys, par_apf, δ)

figure(1)
clf()
subplot(2, 1, 1)
plot_data(y[:], Z_ebpf[:], δ=δ, nofig=true)
title("Bootstrap Filter")
legend([L"y_k", L"z_k", L"z_k \pm   \delta"], loc="upper left")
subplot(2, 1, 2)
plot_effective_sample_size(W_ebpf, Γ=Γ_ebpf, nofig=true)
legend([L"N_{eff}"], loc="upper left")

figure(2)
clf()
subplot(2, 1, 1)
title("Auxiliary Filter")
plot_data(y[:], Z_eapf[:], δ=δ, nofig=true)
legend([L"y_k", L"z_k", L"z_k \pm   \delta"], loc="upper left")
subplot(2, 1, 2)
plot_effective_sample_size(W_eapf, Γ=Γ_eapf, nofig=true)
legend([L"N_{eff}"], loc="upper left")

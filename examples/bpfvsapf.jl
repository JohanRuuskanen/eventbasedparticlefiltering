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
N = 1000
T = 200
δ = 4

A = reshape([0.8], (1, 1))
C = reshape([1.0], (1, 1))

nx = size(A, 1)
ny = size(C, 1)

Q = 2*Matrix{Float64}(I, nx, nx)
R = 0.1*Matrix{Float64}(I, ny, ny)

sys = lin_sys_params(A, C, Q, R, T)
x, y = sim_lin_sys(sys)

# For estimation
par1 = pf_params(N, "SOD", N)
par2 = pf_params(N, "MBT", N)

# Using benchmarktools or time?
# Create struct to incorporate output parameters?
X_ebpf1, W_ebpf1, Z_ebpf1, Γ_ebpf1, _, _, _, S_ebpf1 = ebpf(y, sys, par1, δ)
X_eapf1, W_eapf1, Z_eapf1, Γ_eapf1, _, _, _, S_eapf1 = eapf(y, sys, par1, δ)

X_ebpf2, W_ebpf2, Z_ebpf2, Γ_ebpf2, _, _, _, S_ebpf2 = ebpf(y, sys, par2, δ)
X_eapf2, W_eapf2, Z_eapf2, Γ_eapf2, _, _, _, S_eapf2 = eapf(y, sys, par2, δ)

plt[:close]("all")
figsize=(8, 3)
basepath="/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/"
figure(1, figsize=figsize)
clf()
subplot(2, 1, 1)
plot_data(y[:], Z_ebpf1[:], δ=δ, nofig=true)
title("Bootstrap Filter using SOD")
legend([L"y_k", L"z_k", L"H_k"], loc="lower left")
subplot(2, 1, 2)
plot_effective_sample_size(W_ebpf1, Γ=Γ_ebpf1, nofig=true)
legend([L"N_{eff}"], loc="lower left")
savefig(basepath*"bpfSOD.svg")


figure(2, figsize=figsize)
clf()
subplot(2, 1, 1)
title("Auxiliary Filter using SOD")
plot_data(y[:], Z_eapf1[:], δ=δ, nofig=true)
legend([L"y_k", L"z_k", L"H_k"], loc="lower left")
subplot(2, 1, 2)
plot_effective_sample_size(W_eapf1, Γ=Γ_eapf1, nofig=true)
legend([L"N_{eff}"], loc="lower left")
savefig(basepath*"apfSOD.svg")


figure(3, figsize=figsize)
clf()
subplot(2, 1, 1)
plot_data(y[:], Z_ebpf2[:], δ=δ, nofig=true)
title("Bootstrap Filter using MBT")
legend([L"y_k", L"z_k", L"H_k"], loc="lower left")
subplot(2, 1, 2)
plot_effective_sample_size(W_ebpf2, Γ=Γ_ebpf2, nofig=true)
legend([L"N_{eff}"], loc="lower left")
savefig(basepath*"bpfMBT.svg")

figure(4, figsize=figsize)
clf()
subplot(2, 1, 1)
title("Auxiliary Filter using MBT")
plot_data(y[:], Z_eapf2[:], δ=δ, nofig=true)
legend([L"y_k", L"z_k", L"H_k"], loc="lower left")
subplot(2, 1, 2)
plot_effective_sample_size(W_eapf2, Γ=Γ_eapf2, nofig=true)
legend([L"N_{eff}"], loc="lower left")
savefig(basepath*"apfMBT.svg")

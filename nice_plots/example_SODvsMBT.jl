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
X_bpf2, W_bpf2, Z_bpf2, Γ_bpf2, Neff_bpf2, res_bpf2, fail_bpf2, S_bpf2 = eapf(y, sys, par2, δ)

figure(1, figsize=(5, 2.5))
clf()
plot_data(y[:], Z_bpf1[:], δ=δ, nofig=true)
title("Using SOD")
legend([L"y_k", L"z_k", L"H_k"], loc="upper left", fontsize="small")
savefig("/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/SOD_example.svg")


figure(2, figsize=(5, 2.5))
clf()
plot_data(y[:], Z_bpf2[:], δ=δ, nofig=true)
title("Using MBT")
legend([L"y_k", L"z_k", L"H_k"], loc="upper left", fontsize="small")
savefig("/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/MBT_example.svg")

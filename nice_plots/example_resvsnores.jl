"""
Example for demonstrating the effects of using BPF versus APF for different
event-kernels.
"""

using JLD
using PyPlot
using Random
using StatsBase
using Distributions
using LinearAlgebra

include("../funcs/misc.jl")
include("../funcs/plotting.jl")
include("../linear_systems/filters.jl")

Random.seed!(2)

# Parameters
N = 10
T = 20
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
par1 = pf_params(N, "MBT", N)
par2 = pf_params(N, "MBT", -1)

# Using benchmarktools or time?
# Create struct to incorporate output parameters?
X, W, S, ~ = bpf(y, sys, par1)
X_nores, W_nores, S_nores, ~ = bpf(y, sys, par2)

figure(1)
clf()
plot_particle_trace(reshape(X[:, 1, 1], N, 1), reshape(S[:, 1], N, 1), x_true=x[:], nofig=true)
#plot_particle_trace(X[:, 1, :], S, x_true=x[:], nofig=true)
title("State")
legend([L"x_k", L"X^i_k"])

figure(2)
clf()
plot_particle_trace(X[:, 1, :], S, x_true=x[:], nofig=true)
title("With resample")
legend([L"x_k", L"X^i_k"])


figure(3)
clf()
plot_particle_trace(X_nores[:, 1, :], S_nores, x_true=x[:], nofig=true)
title("Without resample")
legend([L"x_k", L"X^i_k"])

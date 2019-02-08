"""
Example for demonstrating the effects of using BPF versus APF for different
event-kernels.
"""

using Random
using PyPlot
using LinearAlgebra
using EventBasedParticleFiltering

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
X, W, S, _ = bpf(y, sys, par1)
X_nores, W_nores, S_nores, _ = bpf(y, sys, par2)

figure(1, figsize=(6, 3))
clf()
plot_particle_trace(reshape(X[:, 1, 1], N, 1), reshape(S[:, 1], N, 1), x_true=x[:], nofig=true)
#plot_particle_trace(X[:, 1, :], S, x_true=x[:], nofig=true)
title("State")
legend([L"x_k", L"X^i_k"], loc="upper right")
ylim([-7, 7])
#savefig("/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/test_sample.svg")


# No resample
figure(2, figsize=(6, 3))
clf()
plot_particle_trace(X[:, 1, :], S, x_true=x[:], nofig=true)
title("With resample")
legend([L"x_k", L"X^i_k"], loc="upper right")
ylim([-7, 7])
#savefig("/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/with_sample.svg")


figure(3, figsize=(6, 3))
clf()
plot_particle_trace(X_nores[:, 1, 1:2], S_nores[:, 1:2], x_true=x[:], nofig=true)
title("Without resample")
legend([L"x_k", L"X^i_k"], loc="upper right")
ylim([-7, 7])
#savefig("/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/no_sample2.svg")


figure(4, figsize=(6, 3))
clf()
plot_particle_trace(X_nores[:, 1, 1:3], S_nores[:, 1:3], x_true=x[:], nofig=true)
title("Without resample")
legend([L"x_k", L"X^i_k"], loc="upper right")
ylim([-7, 7])
#savefig("/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/no_sample3.svg")


figure(5, figsize=(6, 3))
clf()
plot_particle_trace(X_nores[:, 1, 1:4], S_nores[:, 1:4], x_true=x[:], nofig=true)
title("Without resample")
legend([L"x_k", L"X^i_k"], loc="upper right")
ylim([-7, 7])
#savefig("/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/no_sample4.svg")


figure(6, figsize=(6, 3))
clf()
plot_particle_trace(X_nores[:, 1, :], S_nores, x_true=x[:], nofig=true)
title("Without resample")
legend([L"x_k", L"X^i_k"], loc="upper right")
ylim([-7, 7])
#savefig("/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/no_sampleall.svg")

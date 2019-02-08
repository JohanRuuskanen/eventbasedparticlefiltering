"""
Example for demonstrating the effects of resampling
"""

using Random
using PyPlot
using LinearAlgebra
using EventBasedParticleFiltering

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
output_bpf1 = ebpf(y, sys, par1, δ)
output_apf1 = eapf(y, sys, par1, δ)

output_bpf2 = ebpf(y, sys, par2, δ)
output_apf2 = eapf(y, sys, par2, δ)

plt[:close]("all")
figsize=(8, 3)
#basepath="/home/johanr/Store/presentations/inspiration_coffe_feb-19/graphics/"
figure(1, figsize=figsize)
clf()
subplot(2, 1, 1)
plot_data(y[:], output_bpf1.Z[:], δ=δ, nofig=true)
title("Bootstrap Filter using SOD")
legend([L"y_k", L"z_k", L"H_k"], loc="lower left")
subplot(2, 1, 2)
plot_effective_sample_size(output_bpf1.W, Γ=output_bpf1.Γ, nofig=true)
legend([L"N_{eff}"], loc="lower left")
#savefig(basepath*"bpfSOD.svg")


figure(2, figsize=figsize)
clf()
subplot(2, 1, 1)
title("Auxiliary Filter using SOD")
plot_data(y[:], output_apf1.Z[:], δ=δ, nofig=true)
legend([L"y_k", L"z_k", L"H_k"], loc="lower left")
subplot(2, 1, 2)
plot_effective_sample_size(output_apf1.W, Γ=output_apf1.Γ, nofig=true)
legend([L"N_{eff}"], loc="lower left")
#savefig(basepath*"apfSOD.svg")


figure(3, figsize=figsize)
clf()
subplot(2, 1, 1)
plot_data(y[:], output_bpf2.Z[:], δ=δ, nofig=true)
title("Bootstrap Filter using MBT")
legend([L"y_k", L"z_k", L"H_k"], loc="lower left")
subplot(2, 1, 2)
plot_effective_sample_size(output_bpf2.W, Γ=output_bpf2.Γ, nofig=true)
legend([L"N_{eff}"], loc="lower left")
#savefig(basepath*"bpfMBT.svg")

figure(4, figsize=figsize)
clf()
subplot(2, 1, 1)
title("Auxiliary Filter using MBT")
plot_data(y[:], output_apf2.Z[:], δ=δ, nofig=true)
legend([L"y_k", L"z_k", L"H_k"], loc="lower left")
subplot(2, 1, 2)
plot_effective_sample_size(output_apf2.W, Γ=output_apf2.Γ, nofig=true)
legend([L"N_{eff}"], loc="lower left")
#savefig(basepath*"apfMBT.svg")


using Random
using PyPlot
using LinearAlgebra
using EventBasedParticleFiltering

Random.seed!(2)

# Parameters
N = 10
T = 200
δ = 4

#A = [0.8 1; 0 0.95]
#C = [0.7 0.6]

A = reshape([1.0], (1, 1))
C = reshape([1.0], (1, 1))

nx = size(A, 1)
ny = size(C, 1)

Q = 1*Matrix{Float64}(I, nx, nx)
R = 0.1*Matrix{Float64}(I, ny, ny)

sys = sys_params(A, C, Q, R, T)
x, y = sim_sys(sys)

# For estimation
par_bpf = pf_params(N, "SOD", N/2)
par_apf = pf_params(N, "SOD", N/2)

# Using benchmarktools or time?
# Create struct to incorporate output parameters?
output_bpf = ebpf(y, sys, par_bpf, δ)
output_apf = eapf(y, sys, par_apf, δ)

figure(1)
clf()
subplot(3, 1, 1)
plot_data(y[:], output_bpf.Z[:], δ=δ, nofig=true)
subplot(3, 1 ,2)
plot_particle_trace(output_bpf.X[:,1,:], output_bpf.S, x_true=x[1,:],
    Γ=output_bpf.Γ, nofig=true)
subplot(3, 1, 3)
plot_effective_sample_size(output_bpf.W, Γ=output_bpf.Γ, nofig=true)

figure(2)
clf()
subplot(3, 1, 1)
plot_data(y[:], output_apf.Z[:], δ=δ, nofig=true)
subplot(3, 1 ,2)
plot_particle_trace(output_apf.X[:,1,:], output_apf.S, x_true=x[1,:],
    Γ=output_apf.Γ, nofig=true)
subplot(3, 1, 3)
plot_effective_sample_size(output_apf.W, Γ=output_apf.Γ, nofig=true)


using Random
using PyPlot
using EventBasedParticleFiltering

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
output_bpf_SOD = ebpf(y, sys, par1, δ)
output_bpf_MBT = ebpf(y, sys, par2, δ)
output_apf_SOD = eapf(y, sys, par1, δ)
output_apf_MBT = eapf(y, sys, par2, δ)

figure(1)
clf()
subplot(2, 2, 1)
plot_data(y[:], output_bpf_SOD.Z[:], δ=δ, nofig=true)
title("Using SOD BPF")
legend([L"y_k", L"z_k", L"z_k \pm   \delta"])
subplot(2, 2, 2)
plot_data(y[:], output_bpf_MBT.Z[:], δ=δ, nofig=true)
title("Using MBT BPF")
legend([L"y_k", L"z_k", L"z_k \pm   \delta"])
subplot(2, 2, 3)
plot_data(y[:], output_apf_SOD.Z[:], δ=δ, nofig=true)
title("Using SOD APF")
legend([L"y_k", L"z_k", L"z_k \pm   \delta"])
subplot(2, 2, 4)
plot_data(y[:], output_apf_MBT.Z[:], δ=δ, nofig=true)
title("Using MBT APF")
legend([L"y_k", L"z_k", L"z_k \pm   \delta"])

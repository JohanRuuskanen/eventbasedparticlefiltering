using JLD
using PyPlot
using StatsBase
using Distributions


include("src/misc.jl")
include("src/event_kernels.jl")
include("src/particle_filters.jl")
include("src/particle_filters_eventbased.jl")

# Parameters
T = 100

# Nonlinear and non-Gaussian system
f(x, t) = MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos(1.2*t), 10*eye(1))
h(x, t) = MvNormal(x.^2/20, 0.1*eye(1)) #MvNormal(atan.(x), 1*eye(1))
nd = [1, 1]

sys = sys_params(f, h, T, nd)
par = pf_params(N)

x, y = sim_sys(sys)

X_nbpf, W_nbpf, xh_nbpf, yh_nbpf, Z_nbpf, Γ_nbpf = ebpf_naive(y, sys, par, δ)
X_zbpf, W_zbpf, xh_zbpf, yh_zbpf, Z_zbpf, Γ_zbpf = ebpf_usez(y, sys, par, δ)

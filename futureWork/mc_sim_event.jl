using PyPlot
using StatsBase
using Distributions

include("src/misc.jl")
include("src/event_kernels.jl")
include("src/particle_filters.jl")
include("src/particle_filters_eventbased.jl")


# Parameters
sims = 100
Δ = 0:1:20
N = 100
T = 1000

srand(1)

# Nonlinear and non-Gaussian system
f(x, t) = MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos.(1.2*t), 10*eye(1))
h(x, t) = MvNormal(x.^2/20, 1*eye(1))
nd = [1, 1]

sys = sys_params(f, h, T, nd)

# For estimation
bpf_par = pf_params(
            (x, t) -> MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos(1.2*t), 10*eye(1)), # proposal
            (x, t) -> 0, # predictive likelihood
            N)

M = length(Δ)

Err_nbpf = zeros(sims, M)
Err_zbpf = zeros(sims, M)

for k = 1:sims
    for i = 1:M

        println("$(k), $(i)")
        x, y = sim_sys(sys)
        X_nbpf, W_nbpf, xh_nbpf, yh_nbpf, Z_nbpf, Γ_nbpf = ebpf_naive(y, sys, bpf_par, Δ[i])
        X_zbpf, W_zbpf, xh_zbpf, yh_zbpf, Z_zbpf, Γ_zbpf = ebpf_usez(y, sys, bpf_par, Δ[i])

        Err_nbpf[k, i] = mean((xh_nbpf - x).^2)
        Err_zbpf[k, i] = mean((xh_zbpf - x).^2)
    end
end

figure(1)
clf()
plot(Δ, mean(Err_nbpf, 1)')
plot(Δ, mean(Err_zbpf, 1)')

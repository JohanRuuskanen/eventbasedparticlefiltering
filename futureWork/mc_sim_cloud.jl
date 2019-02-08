"""
Cloud setup
"""

W = 23

if length(procs()) == 1
    addprocs(W)
end

using JLD
using StatsBase
using Distributions

@everywhere using StatsBase
@everywhere using Distributions

@everywhere include("/home/johanr/projects/EBPF/src/misc.jl")
@everywhere include("/home/johanr/projects/EBPF/src/event_kernels.jl")
@everywhere include("/home/johanr/projects/EBPF/src/particle_filters.jl")
@everywhere include("/home/johanr/projects/EBPF/src/particle_filters_eventbased.jl")

@everywhere function evaluate_filters(index, δ, N)
    println(index)

    par = pf_params(N)

    x, y = sim_sys(sys)
    X_nbpf, W_nbpf, xh_nbpf, yh_nbpf, Z_nbpf, Γ_nbpf = ebpf_naive(y, sys, par, δ)
    X_zbpf, W_zbpf, xh_zbpf, yh_zbpf, Z_zbpf, Γ_zbpf = ebpf_usez(y, sys, par, δ)

    xh_zbpf2 = zeros(nd[1], T)
    for k = 1:nd[1]
        xh_zbpf2[k, :] = sum(diag(W_zbpf'*X_zbpf[:, k, :]), 2)
    end

    err_nbpf = (x - xh_nbpf).^2
    err_zbpf = (x - xh_zbpf).^2
    err_zbpf2 = (x - xh_zbpf).^2


    return [mean(err_nbpf), mean(err_nbpf[find(x -> x == 1, Γ_nbpf)]),
            mean(err_zbpf), mean(err_zbpf[find(x -> x == 1, Γ_zbpf)]),
            mean(err_zbpf2[find(x -> x == 0, Γ_zbpf)])]
end

@everywhere T = 1000

# Nonlinear and non-Gaussian system
@everywhere f(x, t) = MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos.(1.2*t), 10*eye(1))
@everywhere h(x, t) = MvNormal(x.^2/20, 1*eye(1))
@everywhere nd = [1, 1]

@everywhere sys = sys_params(f, h, T, nd)

"""
Run simulation
"""

Δ = 0:1:20
N = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]
sims = 1000

err_nbpf = zeros(sims, length(Δ), length(N))
err_zbpf = zeros(sims, length(Δ), length(N))
err_nbpf_trigg = zeros(sims, length(Δ), length(N))
err_zbpf_trigg = zeros(sims, length(Δ), length(N))
err_zbpf_notrigg = zeros(sims, length(Δ), length(N))

for k = 1:length(Δ)
    for i = 1:length(N)
        println("====== Δ: $(Δ[k]), N: $(N[i]) ======")

        all_results = pmap(1:sims) do index
            result = evaluate_filters(index, Δ[k], N[i])
        end

        for j = 1:sims
            err_nbpf[j, k, i] = all_results[j][1]
            err_nbpf_trigg[j, k, i] = all_results[j][2]

            err_zbpf[j, k, i] = all_results[j][3]
            err_zbpf_trigg[j, k, i] = all_results[j][4]

            err_zbpf_notrigg[j, k, i] = all_results[j][5]
        end

    end
end

save("data_sim.jld", "err_nbpf", err_nbpf, "err_nbpf_trigg", err_nbpf_trigg,
            "err_zbpf", err_zbpf, "err_zbpf_trigg", err_zbpf_trigg,
            "err_zbpf_notrigg", err_zbpf_notrigg)

println("Monte-Carlo simulation complete")

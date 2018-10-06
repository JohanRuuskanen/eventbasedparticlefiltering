
ENV["JULIA_PKGDIR"] = "/var/tmp/johanr/.julia"

Pkg.init()
Pkg.add("StatsBase")
Pkg.add("Distributions")
Pkg.add("JLD")

using JLD
using StatsBase
using Distributions

#Add workers
W = 23
addprocs(W)

@everywhere begin
    include("../src/misc.jl")
    include("filters.jl")
    include("filters_eventbased.jl")

    # Parameters
    T = 1000

    A = [0.8 1; 0 0.95]
    C = [0.7 0.6]

    Q = 0.1*eye(2)
    R = 0.01*eye(1)

    function run_filters(idx, N, δ)

        print("Running sim: $(idx[1]) $(idx[2]) $(idx[3])\n")

        sys = lin_sys_params(A, C, Q, R, T)
        x, y = sim_lin_sys(sys)

        nx = size(A, 1)
        ny = size(C, 1)

        # For estimation
        par = pf_params(N)

        # Eventbased implementations
        X_ebpf, W_ebpf, Z_ebpf, Γ_ebpf = ebpf(y, sys, par, δ)
        X_eapf, W_eapf, Z_eapf, Γ_eapf = eapf(y, sys, par, δ)

        if idx[1] == 1 && idx[2] == 1
            xh_kal, P_kal = kalman_filter(y, sys)
            err_kal = x - xh_kal
        else
            err_kal = [-1.0]
        end

        if idx[1] == 1
            xh_ebse, P_ebse, Z_ebse, Γ_ebse = ebse(y, sys, δ)
            err_ebse = x - xh_ebse
        else
            err_ebse = [-1.0]
            Γ_ebse = [-1.0]
        end

        xh_ebpf = zeros(nx, T)
        xh_eapf = zeros(nx, T)

        for k = 1:nx
            xh_ebpf[k, :] = sum(diag(W_ebpf'*X_ebpf[:, k, :]), 2)
            xh_eapf[k, :] = sum(diag(W_eapf'*X_eapf[:, k, :]), 2)
        end

        err_ebpf = x - xh_ebpf
        err_eapf = x - xh_eapf

        return Dict{String,Array{Float64}}(
                    "err_ebpf" => err_ebpf,
                    "err_eapf" => err_eapf,
                    "err_ebse" => err_ebse,
                    "trig_ebpf" => Γ_ebpf,
                    "trig_eapf" => Γ_eapf,
                    "trig_ebse" => Γ_ebse,
                    "err_kal" => err_kal)
    end
end

"""
Run simulation
"""
N = [500]
Δ = [0 0.8 1.6 2.4 3.2 4.0]
sims = 100

path = "/var/tmp/johanr/"
folder = "/data"

if !isdir(path*folder)
        mkdir(path*folder)
end

for n = 1:length(N)
    for d = 1:length(Δ)

        all_results = pmap(1:sims) do index
            result = run_filters([n, d, index], N[n], Δ[d])
        end

        filename = "/sim_" * string(n) * "_" * string(d) * ".jld"
        save(path*folder*filename, "results", all_results)

    end
end
println("Monte-Carlo simulation complete!")

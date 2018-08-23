ENV["JULIA_PKGDIR"] = "/var/tmp/johanr/.julia"

Pkg.init()
Pkg.update()

Pkg.add("StatsBase")
Pkg.add("Distributions")
Pkg.add("JLD")

using JLD
using StatsBase
using Distributions

# Add workers
W = 12

rmprocs(procs())
addprocs([("cloud-01", W)], topology=:master_slave, exeflags="--compilecache=no", tunnel=true)
#addprocs(W)

@everywhere include("/home/johanr/projects/CloudParallel/install_packages.jl")

# Add common constants and functions
@everywhere begin
    include("/home/johanr/projects/EBPF/src/misc.jl")
    include("/home/johanr/projects/EBPF/test_linear/filters_eventbased.jl")

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

        X_ebpf, W_ebpf, xh_ebpf, yh_ebpf, Z_ebpf, Γ_ebpf = ebpf(y, sys, par, δ)
        X_eapf, W_eapf, xh_eapf, yh_eapf, Z_eapf, Γ_eapf = eapf(y, sys, par, δ)

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
                    "trig_ebpf" => Γ_ebpf,
                    "trig_eapf" => Γ_eapf)
    end
end

"""
Run simulation
"""

N = [10 20 30 40 50 60 70 80 90 100]
Δ = [0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6 4.0]
sims = 1000

E = Array{Any}(length(N), length(Δ), sims)
for n = 1:length(N)
    for d = 1:length(Δ)

        all_results = pmap(1:sims) do index
        end 
            result = run_filters([n, d, index], N[n], Δ[d])

        for k = 1:sims
            E[n, d, k] = all_results[k]
        end

    end
end

save("/home/johanr/projects/EBPF/data/mc_data1.jld", "results", E)
println("Monte-Carlo simulation complete!")

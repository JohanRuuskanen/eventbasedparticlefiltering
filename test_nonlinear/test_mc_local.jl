
ENV["JULIA_PKGDIR"] = "/var/tmp/johanr/.julia"

Pkg.init()
Pkg.add("StatsBase")
Pkg.add("Distributions")
Pkg.add("JLD")

using JLD
using StatsBase
using Distributions

#Add workers
W = 12
addprocs(W)

@everywhere begin
    using StatsBase
    using Distributions

    include("/var/tmp/johanr/eb_apf/src/misc.jl")
    include("/var/tmp/johanr/eb_apf/test_nonlinear/filters.jl")
    include("/var/tmp/johanr/eb_apf/test_nonlinear/filters_eventbased.jl")

    # Parameters
    T = 1000

    w = Normal(0.0, 1)
    v = Normal(0.0, 0.1)

    f(x, k) = x/2 + 25*x./(1 + x.^2) + 8*cos(1.2*k)
    h(x, k) = x.^2/20

    sys = sys_params(f, h, w, v, T, [1, 1])

    function run_filters(idx, N, δ)

        print("Running sim: $(idx[1]) $(idx[2]) $(idx[3])\n")

        x, y = sim_sys(sys)
        par = pf_params(N)

        # Eventbased implementations
        X_ebpf, W_ebpf, Z_ebpf, Γ_ebpf, Neff_ebpf, res_ebpf, fail_ebpf = ebpf(y, sys, par, δ)
        X_eapf, W_eapf, Z_eapf, Γ_eapf, Neff_eapf, res_eapf, fail_eapf = eapf(y, sys, par, δ)

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
                    "trig_eapf" => Γ_eapf,
                    "Neff_ebpf" => Neff_ebpf,
                    "Neff_eapf" => Neff_eapf,
                    "res_ebpf" => res_ebpf,
                    "res_eapf" => res_eapf,
                    "fail_ebpf" => fail_ebpf,
                    "fail_eapf" => fail_eapf
                   )

    end
end

"""
Run simulation
"""
N = [200]
Δ = [1 1 2 3 4 5]
sims = 101

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


ENV["JULIA_PKGDIR"] = "/var/tmp/johanr/.julia"

Pkg.init()
Pkg.add("StatsBase")
Pkg.add("ForwardDiff")
Pkg.add("Distributions")
Pkg.add("JLD")

using JLD
using StatsBase
using ForwardDiff
using Distributions

#Add workers
W = 24
addprocs(W)

@everywhere begin
    using StatsBase
    using ForwardDiff
    using Distributions

    include("/var/tmp/johanr/eb_apf/test_nonlinear/bearing/funcs.jl")
    include("/var/tmp/johanr/eb_apf/test_nonlinear/bearing/filters_eventbased.jl")

    # Parameters
    T = 1000

    w = MvNormal([0.0], 1*eye(1))
    v = MvNormal([0.0], sqrt(0.1)*eye(1))

    f(x, k) = x/2 + 25*x./(1 + x.^2) + 8*cos(1.2*k)
    h(x, k) = x.^2/20

    x0 = [0]
    X0 = Normal(0, 10)
    sys = sys_params(f, h, w, v, T, [1, 1])

    function run_filters(idx, N, δ)

        print("Running sim: $(idx[1]) $(idx[2]) $(idx[3])\n")

        x, y = sim_sys(sys, x0)
        par = pf_params(N, X0)

        # Eventbased implementations
        X_ebpf, W_ebpf, Z_ebpf, Γ_ebpf, Neff_ebpf, res_ebpf, fail_ebpf = ebpf_mbt(y, sys, par, δ)
        X_eapf, W_eapf, Z_eapf, Γ_eapf, Neff_eapf, res_eapf, fail_eapf = eapf_mbt(y, sys, par, δ)

        xh_ebpf = zeros(1, T)
        xh_eapf = zeros(1, T)

        for k = 1:1
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
N = [10, 100] #[250]
Δ = [0, 1] #[0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0]
sims = 50 #1000

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

ENV["JULIA_PKGDIR"] = "/var/tmp/johanr/.julia"

Pkg.init()

Pkg.add("StatsBase")
Pkg.add("Distributions")
Pkg.add("JLD")

using JLD
using StatsBase
using Distributions

# Add workers
W = 3
hosts =  vcat([(@sprintf("philon-%2.2d", i), W) for i in 1:12],
       [(@sprintf("heron-%2.2d", i), W) for i in 1:12])
if nprocs() == 1
    #@show addprocs([("heron-01", W)], topology=:master_slave,
    #        exeflags="--compilecache=no", tunnel=true)
    @show addprocs(hosts, topology=:master_slave, exeflags="--compilecache=no", tunnel=true)
end

function sinclude(path)
    open(path) do f
        text = readstring(f)
        s    = 1
        while s <= length(text)
            ex, s = parse(text, s)
            @everywhere @eval $ex
        end
    end
end
#addprocs(W, exeflags="--compilecache=no")

#@everywhere include("/home/johanr/projects/CloudParallel/install_packages.jl")
sinclude("/home/johanr/projects/EBPF/test_linear/testinclude.jl")
sinclude("/home/johanr/projects/EBPF/src/misc.jl")
sinclude("/home/johanr/projects/EBPF/test_linear/filters.jl")
sinclude("/home/johanr/projects/EBPF/test_linear/filters_eventbased.jl")
f = @spawnat 3 isdefined(:fix_sym)
fetch(f)
# Add common constants and functions
@everywhere begin

    using StatsBase
    using Distributions

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
        X_ebpf, W_ebpf, xh_ebpf, yh_ebpf, Z_ebpf, Γ_ebpf = ebpf(y, sys, par, δ)
        X_eapf, W_eapf, xh_eapf, yh_eapf, Z_eapf, Γ_eapf = eapf(y, sys, par, δ)

        # Normal implementations
        X_bpf, W_bpf = bpf(y, sys, par)
        X_apf, W_apf = apf(y, sys, par)

        xh_ebpf = zeros(nx, T)
        xh_eapf = zeros(nx, T)
        xh_bpf = zeros(nx, T)
        xh_apf = zeros(nx, T)
        for k = 1:nx
            xh_ebpf[k, :] = sum(diag(W_ebpf'*X_ebpf[:, k, :]), 2)
            xh_eapf[k, :] = sum(diag(W_eapf'*X_eapf[:, k, :]), 2)
            xh_bpf[k, :] = sum(diag(W_bpf'*X_bpf[:, k, :]), 2)
            xh_apf[k, :] = sum(diag(W_apf'*X_apf[:, k, :]), 2)
        end

        err_ebpf = x - xh_ebpf
        err_eapf = x - xh_eapf
        err_bpf = x - xh_bpf
        err_apf = x - xh_apf

        return Dict{String,Array{Float64}}(
                    "err_ebpf" => err_ebpf,
                    "err_eapf" => err_eapf,
                    "err_bpf" => err_bpf,
                    "err_apf" => err_apf)
    end
end

"""
Run simulation
"""

N = [10 50 100 150 200 250 300 350 400 450 500 600 700 800 900 1000 2000 3000 5000]
Δ = [0]
sims = 100

path = "/home/johanr/projects/EBPF/data"
folder = "/test_results_over_N"

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

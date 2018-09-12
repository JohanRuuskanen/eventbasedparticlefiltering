
ENV["JULIA_PKGDIR"] = "/var/tmp/johanr/.julia2"

Pkg.init()

Pkg.add("StatsBase")
Pkg.add("Distributions")
Pkg.add("JLD")

using JLD
using StatsBase
using Distributions

# Add workers
W = 23
#hosts =  vcat([(@sprintf("philon-%2.2d", i), W) for i in 1:12],
#      [(@sprintf("heron-%2.2d", i), W) for i in 1:12])
hosts = [(@sprintf("cloud-%2.2d", i), W) for i in 1:1]
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
f = @spawnat 8 isdefined(:fix_sym)
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

    function run_filters(idx, δ)
        print("Running sim: $(idx[1]) $(idx[2])\n")

        sys = lin_sys_params(A, C, Q, R, T)
        x, y = sim_lin_sys(sys)

        if δ == 0
            xh_kal, P_kal = kalman_filter(y, sys)
            err_kal = x - xh_kal
        else
            err_kal = [-1.0]
        end

        xh_ebse, P_ebse, Z_ebse, Γ_ebse = ebse(y, sys, δ)
        err_ebse = x - xh_ebse

        return Dict{String,Array{Float64}}(
                    "err_ebse" => err_ebse,
                    "trig_ebse" => Γ_ebse,
                    "err_kal" => err_kal)

    end

end

"""
Run simulation
"""
Δ = [0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.8 4.0]
sims = 1000

path = "/home/johanr/projects/EBPF/test_linear/data"
folder = "test_linear_system_ref"

if !isdir(path*folder)
        mkdir(path*folder)
end

for i = 1:length(Δ)

    all_results = pmap(1:sims) do index
        result = run_filters([i, index], Δ[i])
    end

    filename = "/sim_" * string(i) * ".jld"
    save(path*folder*filename, "results", all_results)

end
println("Monte-Carlo simulation complete!")

"""
Cloud setup
"""

W = 7

if length(procs()) == 1
    addprocs(W)
end

@everywhere using StatsBase
@everywhere using Distributions

@everywhere include("/local/home/johanr/projects/EBPF/src/misc.jl")
@everywhere include("/local/home/johanr/projects/EBPF/src/event_kernels.jl")
@everywhere include("/local/home/johanr/projects/EBPF/src/particle_filters.jl")

@everywhere function evaluate_filters(index)
    println(index)

    x, y = sim_sys(sys)
    X_bpf, W_bpf = bpf(y, sys, par)
    X_sir, W_sir = sir(y, sys, par)
    X_apf, W_apf = apf(y, sys, par)

    xh_bpf = sum(diag(W_bpf'*X_bpf[:, 1, :]), 2)
    xh_sir = sum(diag(W_sir'*X_sir[:, 1, :]), 2)
    xh_apf = sum(diag(W_apf'*X_apf[:, 1, :]), 2)

    return [sqrt(mean((xh_bpf - x').^2)), sqrt(mean((xh_sir - x').^2)), sqrt(mean((xh_apf - x').^2))]
end

@everywhere N = 100
@everywhere T = 100

# Nonlinear and non-Gaussian system
@everywhere f(x, t) = MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos.(1.2*t), 10*eye(1))
@everywhere h(x, t) = MvNormal(x.^2/20, 1*eye(1))
@everywhere nd = [1, 1]

@everywhere sys = sys_params(f, h, T, nd)

# For estimation
@everywhere par = pf_params(N)

"""
Run simulation
"""

sims = 100
all_results = pmap(1:sims) do index
    result = evaluate_filters(index)
end

Err_bpf = zeros(0)
Err_sir = zeros(0)
Err_apf = zeros(0)
for k = 1:length(all_results)
    if !isnan(all_results[k][1])
        append!(Err_bpf, all_results[k][1])
    end

    if !isnan(all_results[k][2])
        append!(Err_sir, all_results[k][2])
    end

    if !isnan(all_results[k][3])
        append!(Err_apf, all_results[k][3])
    end
end

println("")
println("Length: $(length(Err_bpf)), $(length(Err_sir)), $(length(Err_apf))")
println("")
println("BPF: $(mean(Err_bpf)) +- $(std(Err_bpf))")
println("SIR: $(mean(Err_sir)) +- $(std(Err_sir))")
println("APF: $(mean(Err_apf)) +- $(std(Err_apf))")
println("")

using Random
using PyPlot
using Distributions
using LinearAlgebra
using EventBasedParticleFiltering

const EP = EventBasedParticleFiltering

## 1

Random.seed!(2)

D = 3
N = 100
T = 10000

sys_type = "nonlinear_classic"
sys, A, C, Q, E, p0 = EP.generate_example_system(T, type=sys_type)
q, qv = EP.genprop_linear_gaussian_noise()
x, y = sim_sys(sys, x0=rand(p0))

println("Running normal")
opt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([7.5]),
                   pftype=pftype_auxiliary(q=q, qv=qv, D=D))
output = ebpf(y, opt, X0=rand(p0, N)')

res = compute_err_metrics(output, x)

println("Running predcomp")
opt_precomp = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([7.5]),
                   pftype=pftype_auxiliary(q=q, qv=qv, D=D),
                   predictive_computation=true,
                   extra_params=Dict("a" => 0.80))
output_precomp= ebpf(y, opt_precomp, X0=rand(p0, N)')

res_precomp = compute_err_metrics(output_precomp, x)


function print_results(res, text)
  println("$text t/nT/Tf: $(round(norm(res.err_trig, Inf), digits=3)), $(round(norm(res.err_noTrig, Inf), digits=3)), $(res.nbr_frac)")
  println("$text ceT/ceNT/ceErr/Tf: $(round(norm(res.ce_trig, Inf), digits=3)), $(round(norm(res.ce_noTrig, Inf), digits=3)), $(res.ce_errs), $(res.nbr_frac)")
end

println("=====")
print_results(res, "Normal")
print_results(res_precomp, "Precomp")

t = output_precomp.extra["triggerwhen"]
println("")
println("Fraction of trigger before")
println("$(1 - sum(t[:, 1] .== t[:, 2]) / size(t, 1))")

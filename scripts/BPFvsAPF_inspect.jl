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
T = 1000

sys_type = "nonlinear_classic"
sys, A, C, Q, E, p0 = EP.generate_example_system(T, type=sys_type)
q, qv = EP.genprop_linear_gaussian_noise()
x, y = sim_sys(sys, x0=rand(p0))

println("Running BPF SOD")
opt_bpf_sod = ebpf_options(sys=sys, N=N, kernel=kernel_SOD([5.0]),
                    pftype=pftype_bootstrap())
output_bpf_sod = ebpf(y, opt_bpf_sod, X0=rand(p0, N)')
res_bpf_sod = compute_err_metrics(output_bpf_sod, x)

println("Running APF SOD")
opt_apf_sod = ebpf_options(sys=sys, N=N, kernel=kernel_SOD([5.0]),
                    pftype=pftype_auxiliary(q=q, qv=qv, D=D))
output_apf_sod = ebpf(y, opt_apf_sod, X0=rand(p0, N)')
res_apf_sod = compute_err_metrics(output_apf_sod, x)

println("Running BPF IBT")
opt_bpf_ibt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([5.0]),
                    pftype=pftype_bootstrap())
output_bpf_ibt = ebpf(y, opt_bpf_ibt, X0=rand(p0, N)')
res_bpf_ibt = compute_err_metrics(output_bpf_ibt, x)

println("Running APF IBT")
opt_apf_ibt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([5.0]),
                    pftype=pftype_auxiliary(q=q, qv=qv, D=D))
output_apf_ibt = ebpf(y, opt_apf_ibt, X0=rand(p0, N)')
res_apf_ibt = compute_err_metrics(output_apf_ibt, x)

function print_results(res, text)
  println("$text t/nT/Tf: $(round(norm(res.err_trig, Inf), digits=3)), $(round(norm(res.err_noTrig, Inf), digits=3)), $(res.nbr_frac)")
  println("$text ceT/ceNT/ceErr/Tf: $(round(norm(res.ce_trig, Inf), digits=3)), $(round(norm(res.ce_noTrig, Inf), digits=3)), $(res.ce_errs), $(res.nbr_frac)")
end

println("=======")
print_results(res_bpf_sod, "BPF SOD")
print_results(res_apf_sod, "APF SOD")
print_results(res_bpf_ibt, "BPF IBT")
print_results(res_apf_ibt, "APF IBT")
## Plotting

# Comment out figure(2) if setting this or N large
t_int = [60, 85]

figure(1)
clf()
subplot(2, 2, 1)
plot_measurement_data(output_bpf_sod, y, nofig=true, t_int=t_int)
title("BPF SOD")
subplot(2, 2, 2)
plot_measurement_data(output_apf_sod, y, nofig=true, t_int=t_int)
title("APF SOD")
subplot(2, 2, 3)
plot_measurement_data(output_bpf_ibt, y, nofig=true, t_int=t_int)
title("BPF IBT")
subplot(2, 2, 4)
plot_measurement_data(output_apf_ibt, y, nofig=true, t_int=t_int)
title("APF IBT")

figure(2)
clf()
subplot(2, 2, 1)
plot_particle_trace(output_bpf_sod, x_true=x, nofig=true, t_int=t_int)
title("BPF SOD")
subplot(2, 2, 2)
plot_particle_trace(output_apf_sod, x_true=x, nofig=true, t_int=t_int)
title("APF SOD")
subplot(2, 2, 3)
plot_particle_trace(output_bpf_ibt, x_true=x, nofig=true, t_int=t_int)
title("BPF IBT")
subplot(2, 2, 4)
plot_particle_trace(output_apf_ibt, x_true=x, nofig=true, t_int=t_int)
title("APF IBT")

figure(3)
clf()
subplot(2, 2, 1)
plot_particle_hist(output_bpf_sod, x_true=x, nofig=true, t_int=t_int)
title("BPF SOD")
subplot(2, 2, 2)
plot_particle_hist(output_apf_sod, x_true=x, nofig=true, t_int=t_int)
title("APF SOD")
subplot(2, 2, 3)
plot_particle_hist(output_bpf_ibt, x_true=x, nofig=true, t_int=t_int)
title("BPF IBT")
subplot(2, 2, 4)
plot_particle_hist(output_apf_ibt, x_true=x, nofig=true, t_int=t_int)
title("APF IBT")

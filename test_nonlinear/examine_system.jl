
using PyPlot
using Distributions

include("filters.jl")
include("filters_eventbased.jl")
include("../src/misc.jl")

T = 10000
N = 200

δ = 10

w = Normal(0.0, 10)
v = Normal(0.0, 1)


f(x, k) = x/2 + 25*x./(1 + x.^2) + 8*cos(1.2*k)
h(x, k) = x.^2/20

sys = sys_params(f, h, w, v, T, [1, 1])
par = pf_params(N)

x, y = sim_sys(sys)

#=
xh, Z, Γ = test_mbt(x, y, sys, par, δ) 
idx = find(x->x==1,Γ)

figure(1)
subplot(2, 1, 1)
plot(1:T, y')
plot(1:T, Z')
plot(idx, y[idx], "C2o")
subplot(2, 1, 2)
plot(x')
plot(xh')
=#
tic()
X1, W1, Z1, Γ1, Neff1, res1, fail1 = ebpf(y, sys, par, 5)
t1 = toc()

tic()
X2, W2, Z2, Γ2, Neff2, res2, fail2 = ebpf_mbt(y, sys, par, 5)
t2 = toc()

tic()
X3, W3, Z3, Γ3, Neff3, res3, fail3 = eapf(y, sys, par, 5)
t3 = toc()

tic()
X4, W4, Z4, Γ4, Neff4, res4, fail4 = eapf_mbt(y, sys, par, 5)
t4 = toc()

xh1 = sum(diag(W1'*X1[:, 1, :]), 2)
xh2 = sum(diag(W2'*X2[:, 1, :]), 2)
xh3 = sum(diag(W3'*X3[:, 1, :]), 2)
xh4 = sum(diag(W4'*X4[:, 1, :]), 2)

err1 = x' - xh1
err2 = x' - xh2
err3 = x' - xh3
err4 = x' - xh4

idx1 = find(x -> x == 1, Γ1)
idx2 = find(x -> x == 1, Γ2)
idx3 = find(x -> x == 1, Γ3)
idx4 = find(x -> x == 1, Γ4)

err1_t = err1[idx1]
err2_t = err2[idx2]
err3_t = err3[idx3]
err4_t = err4[idx4]

println("")
println("Total error")
println("BPF_SOD x: $(mean(err1.^2))")
println("BPF_MBT x: $(mean(err2.^2))")
println("APF_SOD x: $(mean(err3.^2))")
println("APF_MBT x: $(mean(err4.^2))")
println("")
println("Error at new measurements")
println("BPF_SOD x: $(mean(err1_t.^2))")
println("BPF_MBT x: $(mean(err2_t.^2))")
println("APF_SOD x: $(mean(err3_t.^2))")
println("APF_MBT x: $(mean(err4_t.^2))")
println("")
println("BPF_SOD t: $(t1)")
println("BPF_MBT t: $(t2)")
println("APF_SOD t: $(t3)")
println("APF_MBT t: $(t4)")
#=
println("")
println("Special")
println("EBPF Neff: $(mean(Neff1))")
println("EBPF res: $(sum(res1))")
println("EBPF fail: $(sum(fail1))")
println("EBPF trig: $(sum(Γ1))")
println("")
println("EAPF Neff: $(mean(Neff2))")
println("EAPF res: $(sum(res2))")
println("EAPF fail: $(sum(fail2))")
println("EAPF trig: $(sum(Γ2))")
println("")

figure(1)
clf()
subplot(2, 1, 1)
plot(y')
plot(Z1')
plot(Z2')
legend(["y", "z BPF", "z APF"])
subplot(2, 1, 2)
plot(x')
plot(xh1)
plot(xh2)
legend(["True", "EBPF", "EAPF"])

idx_res1 = find(x->x == 1, res1)
idx_fail1 = find(x->x == 1, fail1)
idx_res2 = find(x->x == 1, res2)
idx_fail2 = find(x->x == 1, fail2)
figure(2)
clf()
title("Effective sample size")
subplot(2, 1, 1)
plot(1:T, Neff1, "C0")
plot((1:T)[idx_res1], Neff1[idx_res1], "C0o")
plot((1:T)[idx_fail1], Neff1[idx_fail1], "C0x")
subplot(2, 1, 2)
plot(1:T, Neff2, "C0")
plot((1:T)[idx_res2], Neff2[idx_res2], "C0o")
plot((1:T)[idx_fail2], Neff2[idx_fail2], "C0x")
=#

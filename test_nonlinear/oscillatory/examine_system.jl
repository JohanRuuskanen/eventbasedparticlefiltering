
using PyPlot
using Distributions

include("filters.jl")
include("filters_eventbased.jl")
include("../../src/misc.jl")

T = 1000
N = 200

δ = 10

w = Normal(0.0, 1)
v = Normal(0.0, 0.1)

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
#tic()
#X1, W1, Z1, Γ1, Neff1, res1, fail1 = ebpf(y, sys, par, 0)
#t1 = toc()

tic()
X2, W2, Z2, Γ2, Neff2, res2, fail2 = ebpf_mbt(y, sys, par, 2)
t2 = toc()

#tic()
#X2, W2, Z2, Γ2, Neff2, res2, fail2 = eapf(y, sys, par, 0)
#t3 = toc()

tic()
X4, W4, Z4, Γ4, Neff4, res4, fail4 = eapf_mbt(y, sys, par, 2)
t4 = toc()

#xh1 = sum(diag(W1'*X1[:, 1, :]), 2)
xh2 = sum(diag(W2'*X2[:, 1, :]), 2)
#xh3 = sum(diag(W3'*X3[:, 1, :]), 2)
xh4 = sum(diag(W4'*X4[:, 1, :]), 2)

#err1 = x' - xh1
err2 = x' - xh2
#err3 = x' - xh3
err4 = x' - xh4

#idx1 = find(x -> x == 1, Γ1)
idx2 = find(x -> x == 1, Γ2)
#idx3 = find(x -> x == 1, Γ3)
idx4 = find(x -> x == 1, Γ4)

#err1_t = err1[idx1]
err2_t = err2[idx2]
#err3_t = err3[idx3]
err4_t = err4[idx4]

println("")
println("Total error")
#println("BPF_SOD x: $(mean(err1.^2))")
println("BPF_MBT x: $(mean(err2.^2))")
#println("APF_SOD x: $(mean(err3.^2))")
println("APF_MBT x: $(mean(err4.^2))")
println("")
println("Error at new measurements")
#println("BPF_SOD x: $(mean(err1_t.^2))")
println("BPF_MBT x: $(mean(err2_t.^2))")
#println("APF_SOD x: $(mean(err3_t.^2))")
println("APF_MBT x: $(mean(err4_t.^2))")
println("")
#println("BPF_SOD t: $(t1)")
println("BPF_MBT t: $(t2)")
#println("APF_SOD t: $(t3)")
println("APF_MBT t: $(t4)")
println("")
println("Special")
#println("EBPF_SOD Neff: $(mean(Neff1))")
#println("EBPF_SOD res: $(sum(res1))")
#println("EBPF_SOD fail: $(sum(fail1))")
#println("EBPF_SOD trig: $(sum(Γ1))")
#println("")
println("EBPF_MBT Neff: $(mean(Neff2))")
println("EBPF_MBT res: $(sum(res2))")
println("EBPF_MBT fail: $(sum(fail2))")
println("EBPF_MBT trig: $(sum(Γ2))")
println("")
#println("EAPF_SOD Neff: $(mean(Neff3))")
#println("EAPF_SOD res: $(sum(res3))")
#println("EAPF_SOD fail: $(sum(fail3))")
#println("EAPF_SOD trig: $(sum(Γ3))")
#println("")
println("EAPF_MBT Neff: $(mean(Neff4))")
println("EAPF_MBT res: $(sum(res4))")
println("EAPF_MBT fail: $(sum(fail4))")
println("EAPF_MBT trig: $(sum(Γ4))")
println("")

figure(1)
clf()
subplot(2, 1, 1)
plot(y')
#plot(Z1')
plot(Z2')
#plot(Z3')
plot(Z4')
legend(["y", "z BPF_SOD", "z BPF_MBT", "z APF_SOD", "Z APF_MBT"])
subplot(2, 1, 2)
plot(x')
#plot(xh1)
plot(xh2)
#plot(xh3)
plot(xh4)
legend(["True", "BPF_SOD", "BPF_MBT", "APF_SOD", "APF_MBT"])

#idx_res1 = find(x->x == 1, res1)
#idx_fail1 = find(x->x == 1, fail1)
idx_res2 = find(x->x == 1, res2)
idx_fail2 = find(x->x == 1, fail2)
#idx_res3 = find(x->x == 1, res3)
#idx_fail3 = find(x->x == 1, fail3)
idx_res4 = find(x->x == 1, res4)
idx_fail4 = find(x->x == 1, fail4)
figure(2)
clf()
title("Effective sample size")
#subplot(4, 1, 1)
#plot(1:T, Neff1, "C0")
#plot((1:T)[idx_res1], Neff1[idx_res1], "C0o")
#plot((1:T)[idx_fail1], Neff1[idx_fail1], "C0x")
subplot(4, 1, 2)
plot(1:T, Neff2, "C0")
plot((1:T)[idx_res2], Neff2[idx_res2], "C0o")
plot((1:T)[idx_fail2], Neff2[idx_fail2], "C0x")
#subplot(4, 1, 3)
#plot(1:T, Neff3, "C0")
#plot((1:T)[idx_res3], Neff3[idx_res3], "C0o")
#plot((1:T)[idx_fail3], Neff3[idx_fail3], "C0x")#
subplot(4, 1, 4)
plot(1:T, Neff4, "C0")
plot((1:T)[idx_res4], Neff4[idx_res4], "C0o")
plot((1:T)[idx_fail4], Neff4[idx_fail4], "C0x")


using PyPlot
using Distributions

include("filters.jl")
include("filters_eventbased.jl")
include("../src/misc.jl")

T = 1000
N = 200

δ = 5

w = Normal(0.0, 1)
v = Normal(0.0, 0.1)


f(x, k) = x/2 + 25*x./(1 + x.^2) + 8*cos(1.2*k)
h(x, k) = x.^2/20

sys = sys_params(f, h, w, v, T, [1, 1])
par = pf_params(N)

x, y = sim_sys(sys)

tic()
X1, W1, Z1, Γ1, Neff1, res1, fail1 = ebpf(y, sys, par, δ)
t1 = toc()

tic()
X2, W2, Z2, Γ2, Neff2, res2, fail2 = eapf(y, sys, par, δ)
t2 = toc()

xh1 = sum(diag(W1'*X1[:, 1, :]), 2)
xh2 = sum(diag(W2'*X2[:, 1, :]), 2)

err1 = x' - xh1
err2 = x' - xh2

idx1 = find(x -> x == 1, Γ1)
idx2 = find(x -> x == 1, Γ2)

err1_t = err1[idx1]
err2_t = err2[idx2]



println("")
println("Total error")
println("EBPF x: $(mean(err1.^2))")
println("EAPF x: $(mean(err2.^2))")
println("")
println("Error at new measurements")
println("EBPF x: $(mean(err1_t.^2))")
println("EAPF x: $(mean(err2_t.^2))")
println("")
println("EBPF t: $(t1)")
println("EAPF t: $(t2)")
println("")
println("Special")
println("EBPF Neff: $(mean(Neff1))")
println("EBPF res: $(sum(res1))")
println("EBPF fail: $(sum(fail1))")
println("")
println("EAPF Neff: $(mean(Neff2))")
println("EAPF res: $(sum(res2))")
println("EAPF fail: $(sum(fail2))")
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


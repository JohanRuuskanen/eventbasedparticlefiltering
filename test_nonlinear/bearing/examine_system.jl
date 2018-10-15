
using PyPlot
using Distributions

include("filters.jl")
include("filters_eventbased.jl")

struct pf_params
    N::Number
end

struct sys_params
    f
    h
    B
    w
    v
    T
    nd
end

function sim_sys(sys, x0)
    x = zeros(sys.nd[1], sys.T)
    y = zeros(sys.nd[2], sys.T)

    x[:, 1] = x0
    y[:, 1] = sys.h(x0) + rand(sys.v)
    
    BP(x) = exp(5/x) + exp(5/(1000 - x)) - 2

    for k = 2:sys.T
        
        x[:, k] = sys.f(x[:, k-1]) + sys.B*rand(sys.w)

        x[2, k] += sign(500 - x[1, k])*BP(x[1, k]) 
        x[4, k] += sign(500 - x[3, k])*BP(x[3, k]) 

        if x[1, k] < 0; x[1, k] = 1 end
        if x[1, k] > 1000; x[1, k] = 999 end
        if x[3, k] < 0; x[3, k] = 1 end
        if x[3, k] > 1000; x[3, k] = 999 end
        
        y[:, k] = sys.h(x[:, k]) + rand(sys.v)
    end

    return x, y
end

T = 1000
N = 100

A = [1 1 0 0;
     0 1 0 0;
     0 0 1 1;
     0 0 0 1]
B = [1 0;
     0.1 0;
     0 1;
     0 0.1]

Q = 1 * eye(2)
#R = [5 0;
#     0 0.05]
R = [0.05]

w = MvNormal(zeros(2), Q)
v = MvNormal(zeros(1), R)

f(x) = A*x
h(x) = atan(x[1]/x[3])
#h(x) = [sqrt(x[1]^2 + x[3]^2);
#        atan(x[1]/x[3])]

sys = sys_params(f, h, B, w, v, T, [4, 2])
par = pf_params(N)

x0 = [500, 0, 500, 0]
x, y = sim_sys(sys, x0)

#=
figure(1)
clf()
subplot(2, 1, 1)
plot(x[1, :], x[3, :], "C0o")
subplot(2, 1, 2)
plot(y[2, :])

figure(2)
clf()
subplot(2, 1, 1)
plot(x[1, :]./x[3, :])
subplot(2, 1, 1)
plot(atan(x[1, :]./x[3, :]) + rand(v, T)[1, :])
=#

tic()
X1, W1, Z1, Γ1, Neff1, res1, fail1 = ebpf(y, sys, par, 2)
t1 = toc()

#tic()
#X2, W2, Z2, Γ2, Neff2, res2, fail2 = ebpf(y, sys, par, 2)
#t2 = toc()

#tic()
#X3, W3, Z3, Γ3, Neff3, res3, fail3 = eapf(y, sys, par, δ)
#t3 = toc()

#tic()
#X4, W4, Z4, Γ4, Neff4, res4, fail4 = eapf_mbt(y, sys, par, δ)
#t4 = toc()

xh1 = zeros(4, T)

for k = 1:4
    xh1[k, :] = sum(diag(W1'*X1[:, k, :]), 2)
end
#xh2 = sum(diag(W2'*X2[:, 1, :]), 2)
#xh3 = sum(diag(W3'*X3[:, 1, :]), 2)
#xh4 = sum(diag(W4'*X4[:, 1, :]), 2)

err1 = x - xh1
#err2 = x' - xh2
#err3 = x' - xh3
#err4 = x' - xh4

idx1 = find(x -> x == 1, Γ1)
#idx2 = find(x -> x == 1, Γ2)
#idx3 = find(x -> x == 1, Γ3)
#idx4 = find(x -> x == 1, Γ4)

err1_t = err1[idx1]
#err2_t = err2[idx2]
#err3_t = err3[idx3]
#err4_t = err4[idx4]

println("")
println("Total error")
println("BPF_SOD x1: $(mean((err1.^2)[1, :]))")
println("BPF_SOD x2: $(mean((err1.^2)[2, :]))")
println("BPF_SOD x3: $(mean((err1.^2)[3, :]))")
println("BPF_SOD x4: $(mean((err1.^2)[4, :]))")
#println("BPF_MBT x: $(mean(err2.^2))")
#println("APF_SOD x: $(mean(err3.^2))")
#println("APF_MBT x: $(mean(err4.^2))")
#println("")
#println("Error at new measurements")
#println("BPF_SOD x: $(mean(err1_t.^2))")
#println("BPF_MBT x: $(mean(err2_t.^2))")
#println("APF_SOD x: $(mean(err3_t.^2))")
#println("APF_MBT x: $(mean(err4_t.^2))")
#println("")
#println("BPF_SOD t: $(t1)")
#println("BPF_MBT t: $(t2)")
#println("APF_SOD t: $(t3)")
#println("APF_MBT t: $(t4)")
#println("")
println("Special")
println("EBPF_SOD Neff: $(mean(Neff1))")
println("EBPF_SOD res: $(sum(res1))")
println("EBPF_SOD fail: $(sum(fail1))")
println("EBPF_SOD trig: $(sum(Γ1))")
#println("")
#println("EBPF_MBT Neff: $(mean(Neff2))")
#println("EBPF_MBT res: $(sum(res2))")
#println("EBPF_MBT fail: $(sum(fail2))")
#println("EBPF_MBT trig: $(sum(Γ2))")
#println("")
#println("EAPF_SOD Neff: $(mean(Neff3))")
#println("EAPF_SOD res: $(sum(res3))")
#println("EAPF_SOD fail: $(sum(fail3))")
#println("EAPF_SOD trig: $(sum(Γ3))")
#println("")
#println("EAPF_MBT Neff: $(mean(Neff4))")
#println("EAPF_MBT res: $(sum(res4))")
#println("EAPF_MBT fail: $(sum(fail4))")
#println("EAPF_MBT trig: $(sum(Γ4))")
#println("")

y_pred = zeros(2, T)
for k = 1:T
    y_pred[:, k] = h(xh1[:, k])
end

figure(1)
clf()
subplot(3, 1, 1)
plot(x[1, :], x[3, :], "C0o")
axis([0, 1000, 0, 1000])
plot(xh1[1, :], xh1[3, :], "C1x")
subplot(3, 1, 2)
plot(1:T, y[1, :])
plot(1:T, y_pred[1, :])
subplot(3, 1, 3)
plot(1:T, y[2, :])
plot(1:T, y_pred[2, :])


#legend(["y", "z BPF_SOD", "z BPF_MBT", "z APF_SOD", "Z APF_MBT"])
#subplot(2, 1, 2)
#plot(x')
#plot(xh1)
#plot(xh2)
#plot(xh3)
#plot(xh4)
#legend(["True", "BPF_SOD", "BPF_MBT", "APF_SOD", "APF_MBT"])

#idx_res1 = find(x->x == 1, res1)
#idx_fail1 = find(x->x == 1, fail1)
#idx_res2 = find(x->x == 1, res2)
#idx_fail2 = find(x->x == 1, fail2)
#idx_res3 = find(x->x == 1, res3)
#idx_fail3 = find(x->x == 1, fail3)
#idx_res4 = find(x->x == 1, res4)
#idx_fail4 = find(x->x == 1, fail4)
#figure(2)
#clf()
#title("Effective sample size")
#subplot(4, 1, 1)
#plot(1:T, Neff1, "C0")
#plot((1:T)[idx_res1], Neff1[idx_res1], "C0o")
#plot((1:T)[idx_fail1], Neff1[idx_fail1], "C0x")
#subplot(4, 1, 2)
#plot(1:T, Neff2, "C0")
#plot((1:T)[idx_res2], Neff2[idx_res2], "C0o")
#plot((1:T)[idx_fail2], Neff2[idx_fail2], "C0x")
#subplot(4, 1, 3)
#plot(1:T, Neff3, "C0")
#plot((1:T)[idx_res3], Neff3[idx_res3], "C0o")
#plot((1:T)[idx_fail3], Neff3[idx_fail3], "C0x")
#subplot(4, 1, 4)
#plot(1:T, Neff4, "C0")
#plot((1:T)[idx_res4], Neff4[idx_res4], "C0o")
#plot((1:T)[idx_fail4], Neff4[idx_fail4], "C0x")

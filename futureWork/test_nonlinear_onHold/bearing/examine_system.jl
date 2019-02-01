
using PyPlot
using ForwardDiff
using Distributions

include("funcs.jl")
include("filters_eventbased.jl")

 srand(100)
 T = 1000
 N = 250
Δ = [0, 1.0, 2.0, 3.0, 4.0, 5.0]

tests = 10
x0 = [0]
X0 = Normal(0, 10)

nx = 1
ny = 1

Q = 1 * eye(1)
R = sqrt(0.1)*eye(1)

w = MvNormal(zeros(nx), Q)
v = MvNormal(zeros(ny), R)

f(x, k) = x/2 + 25x./(1 + x.^2) + 8*cos(1.2*k)
h(x, k) = x.^2/20

#R = [10 0;
#     0 0.0001]
#R = [500 0;
#     0 1]

#w = MvNormal(zeros(nx), B*Q*B' + 1e-3*eye(nx))
#v = MvNormal(zeros(ny), R)

#f(x) = A*x
#h(x) = [sqrt(x[1]^2 + x[3]^2);
#        atan(x[1]/x[3])]
#h(x) = [sqrt(x[1]^2 + x[3]^2);
#        x[3]]

#hd(x) = [x[1]/sqrt(x[1]^2 + x[3]^2) 0 x[3]/sqrt(x[1]^2 + x[3]^2) 0;
#         x[3]/(x[1]^2 + x[3]^2) 0 -x[1]/(x[1]^2 + x[3]^2) 0]

# Theory works for non-linear systems. However, initial simulations have shown
# it is very dependent on how good the approximation to the optimal proposal
# becomes

sys = sys_params(f, h, w, v, T, [nx, ny])
par = pf_params(N, X0)
x, y = sim_sys(sys, x0)

X = zeros(N, nx, T, tests)
W = zeros(N, T, tests)
Z = zeros(ny, T, tests)
Γ = zeros(T, tests)
Neff = zeros(T, tests)
res = zeros(T, tests)
fail = zeros(T, tests)

tic()
for t = 1:5
    X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = ebpf_mbt(y, sys, par, Δ[t])
end
for t = 6:10
    X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = eapf_mbt(y, sys, par, Δ[t-5])
end
#t = 1
#X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = ebpf(y, sys, par, Δ[5])
#t = 2
#X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = eapf(y, sys, par, Δ[5])
#t = 3
#X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = eapf(y, sys, par, Δ[1])
#t = 4
#X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = eapf(y, sys, par, Δ[2])
#t = 5
#X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = ebpf_mbt(y, sys, par, Δ[t-3])
#t = 6
#X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = ebpf_mbt(y, sys, par, Δ[t-3])
#X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = eapf(y, sys, par, Δ[1])
#X[:, :, :, t], W[:, :, t], Z[:, :, t], Γ[:, t], Neff[:, t], res[:, t], fail[:, t] = ebpf_mbt(y, sys, par, Δ[1])
println("Did iteration")
t1 = toc()

xh = zeros(nx, T, tests)
err = zeros(nx, T, tests)
idx = Array{Any}(tests)
err_t = Array{Any}(tests)
y_p = zeros(ny, T, tests)

for t = 1:tests
    for k = 1:nx
        xh[k, :, t] = sum(diag(W[:, :, t]'*X[:, k, :, t]), 2)
    end
    for k = 1:T
        y_p[:, k, t] = h(xh[:, k, t], k)
    end

    err[:, :, t] = x - xh[:, :, t]
    idx[t] = find(x->x==1, Γ[:, t])
    err_t[t] = err[:, idx[t], t]
end

println("")
println("Total error")
for k = 1:nx
    for t = 1:tests
        println("Test$(t) x$(k): $(mean(err[k, :, t].^2))")
    end
    println("")
end
println("New measurement error")
for k = 1:nx
    for t = 1:tests
        println("Test$(t) x$(k): $(mean((err_t[t])[k, :].^2))")
    end
    println("")
end
println("Special")
for t = 1:tests
    println("Test$(t) Neff: $(mean(Neff[:, t]))")
    println("Test$(t) res: $(sum(res[:, t]))")
    println("Test$(t) fail: $(sum(fail[:, t]))")
    println("Test$(t) trig: $(sum(Γ[:, t]))")
    println("")
end

figure(1)
clf()
for t = 1:tests
    subplot(ny+1, tests, t)
    plot(x[1, :], "C0")
    #axis([0, 1000, 0, 1000])
    plot(xh[1, :, t], "C$(t)")
    for k = 1:ny
        subplot(ny+1, tests, t+k*tests)
        plot(1:T, y[1, :])
        plot(1:T, Z[k, :, t])
        plot(1:T, y_p[k, :, t])
    end
end

figure(2)
clf()
for t = 1:tests
    for k = 1:nx
        subplot(nx, tests, t + (k-1)*nx)
        plot(1:T, x[k, :])
        plot(1:T, xh[k, :, t], "C$(t)")
    end
end

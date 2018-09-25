
using PyPlot
using Distributions

include("filters.jl")
include("../src/misc.jl")

T = 200

N = 100

Q = 1.0*eye(1)
R = 1.0*eye(1)

f(x, k) = MvNormal(x/2 + 25*x./(1 + x.^2) + 8*cos(1*k), Q)
h(x, k) = MvNormal(x.^2/20, R)

sys = sys_params(f, h, Q, R, T, [1, 1])
par = pf_params(N)

x, y = sim_sys(sys)

X1, W1 = bpf(y, sys, N)
X2, W2 = apf(y, sys, par)

xh1 = sum(diag(W1'*X1[:, 1, :]), 2)
xh2 = sum(diag(W2'*X2[:, 1, :]), 2)

figure(1)
clf()
subplot(2, 1, 1)
plot(1:T, y')
subplot(2, 1, 2)
plot(1:T, x')
plot(1:T, xh1)
plot(1:T, xh2)

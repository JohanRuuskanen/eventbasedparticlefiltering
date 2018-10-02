
using PyPlot
using Distributions

include("filters.jl")
include("../src/misc.jl")

T = 200

N = 100

Q = 1.0
R = 0.1

w = Normal(0.0, 1)#Gumbel(0, 1)
v = Normal(0.0, 0.1)

wh = Normal(mean(w), std(w)*sqrt(2))
vh = Normal(mean(v), std(v)*sqrt(2))

f(x, k) = x/2 + 25*x./(1 + x.^2) + 8*cos(1.2*k)
h(x, k) = x.^2/20

sys = sys_params(f, h, w, v, T, [1, 1])
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
legend(["True", "BPF", "APF"])

x = linspace(-10, 10, 1000)
figure(2)
clf()
subplot(2, 1, 1)
plot(x, pdf.(w, x))
plot(x, pdf.(wh, x))
subplot(2, 1, 2)
plot(x, pdf.(v, x))
plot(x, pdf.(vh, x))

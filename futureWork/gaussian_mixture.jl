
using PyPlot
using Distributions

f = Gamma(3, 1)

M = 10
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
V = 1

L = zeros(size(x))
for i = 1:length(x)
    L[i] = pdf.(f, x[i])
end
L /= sum(L)

MG = MixtureModel(map(y -> MvNormal([y...], V), x), L)



x = linspace(-5, 15, 1000)
x_v = [[i] for i in x]

figure(1)
clf()
plot(x, pdf.(f, x))
plot(x, pdf.(MG, x_v))

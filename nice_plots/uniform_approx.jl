using Distributions
using PyPlot

Z = 0
M = 5
δ = 2


mu = linspace(Z-δ, Z+δ, M)
Vn = 2*δ/M * 1/2

MG = MixtureModel(map(x -> Normal(x, Vn), mu))

x = linspace(Z-2*δ, Z+2*δ, 1000)
figure()
clf()
plot(x, pdf.(Uniform(Z - δ, Z + δ), x))
plot(x, pdf(MG, x))

using Distributions
using PyPlot

Z = 0
M = 5
δ = 2


mu = range(Z-δ, stop=Z+δ, length=M)
Vn = 2*δ/M * 1/2

MG = MixtureModel(map(x -> Normal(x, Vn), mu))

x = range(Z-2*δ, stop=Z+2*δ, length=1000)
figure()
clf()
plot(x, pdf.(Uniform(Z - δ, Z + δ), x))
plot(x, pdf(MG, x))

using PyPlot
using Distributions


a = -2
b = 2

U = Uniform(a, b)

N = 20

L = (b - a) / (N-1)

V = L/sqrt(2)

Y = a:L:b

MD = MixtureModel(map(μ -> Normal(μ, V), Y))

x = linspace(1.5*a, 1.5*b, 10000)
figure(1)
clf()
plot(x, pdf.(U, x))
plot(x, pdf.(MD, x), "--")
legend(["Uniform", "Gaussian mix"])

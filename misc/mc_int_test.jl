using PyPlot
using Distributions



pl = Normal(0, 1)
pu = Uniform(0, 1)


# monte carlo integration
N = 10000
X = rand(pu, N)
I = 1/N * sum(pdf.(pl, X))
#println(I)


# ==========
# New test

N = 100000
f(x) = x.^2

d = Uniform(-2, 3)
X = rand(d, N)

I = 1/N * sum(f(X)./pdf.(d, X))
#println(I)


# =======
# New test

D = Gamma(3.6, 2.3)
U = Uniform(-1, 7)

N = 10000
X = rand(U, N)
I1 = 1/N * sum(pdf.(D, X))

I2 = 1/8 * (cdf(D, 7) - cdf(D, -1))

println("$(I1), $(I2)")

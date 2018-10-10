
using Plots
using StatsBase
using Distributions

N = 1000
N2 = 2
M = 20
V = 2/19* 1/sqrt(2)
nx = 10

y0 = -2
y1 = 2

srand(6)

x = ones(N)
y = linspace(y0, y1, 1000)

d1 = MixtureModel(map(x -> Normal(x, V), linspace(-1, 1, M)))
d2 = Normal(1, 0.2)

X1 = rand(d1, nx)
W1 = pdf.(d1, X1)
W2 = pdf.(d2, X1).*W1
W1 /= sum(W1)
W2 /= sum(W2)

idx = rand(Categorical(W1), nx)
idx2 = rand(Categorical(W2), nx)

X2_bpf = X1[idx] + rand(Normal(0, 0.5), nx)
X2_sis = zeros(nx)
X2_apf = zeros(nx) 
for k = 1:nx
    X2_sis[k] = X1[idx[k]] + rand(Normal(0.7*(mean(d2) - X1[idx[k]]), 0.3))
    X2_apf[k] = X1[idx2[k]] + rand(Normal(0.7*(mean(d2) - X1[idx2[k]]), 0.3))
end

pgfplots()

vx = 0
a = 1/pdf(d1, mean(d1))
p = plot(vx*ones(N), y, a*pdf.(d1, y))

push!(p)
#save("test.tex")


"""
figure(1)
clf()
subplot(2, 1, 1)

plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0")    
plot3D(vx*ones(nx), X1, a*pdf.(d1, X1), "C1o")
for k = 1:nx
    plot3D(vx*ones(N2), X1[k]*ones(N2), [0, a*pdf.(d1, X1[k])], "C1:")
end
vx = 1
a = 1/pdf(d2, mean(d2))
plot3D(vx*ones(N), y, a*pdf.(d2, y), "C0")    
plot3D(vx*ones(nx), X2_bpf, a*pdf.(d2, X2_bpf), "C1o")
for k = 1:nx
    plot3D([0, 1], [X1[idx[k]], X2_bpf[k]], [0, 0], "C1:")
    plot3D(vx*ones(N2), X2_bpf[k]*ones(N2), [0, a*pdf.(d2, X2_bpf[k])], "C1:")
end

axis("off")
grid(false)
xlim([-1, 2])
ylim([y0, y1])
show()

figure(2)
clf()
vx = 0
a = 1/pdf(d1, mean(d1))
plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0")    
plot3D(vx*ones(nx), X1, a*pdf.(d1, X1), "C2^")
for k = 1:nx
    plot3D([0, 1], [X1[idx[k]], X2_sis[k]], [0, 0], "C2:")
    plot3D(vx*ones(N2), X1[k]*ones(N2), [0, a*pdf.(d1, X1[k])], "C2:")
end
vx = 1
a = 1/pdf(d2, mean(d2))
plot3D(vx*ones(N), y, a*pdf.(d2, y), "C0")    
plot3D(vx*ones(nx), X2_sis, a*pdf.(d2, X2_sis), "C2^")
for k = 1:nx
    plot3D(vx*ones(N2), X2_sis[k]*ones(N2), [0, a*pdf.(d2, X2_sis[k])], "C2:")
end

axis("off")
grid(false)
xlim([-1, 2])
ylim([y0, y1])

figure(3)
clf()
vx = 0
a = 1/pdf(d1, mean(d1))
plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0")    
plot3D(vx*ones(nx), X1, a*pdf.(d1, X1), "C3x")
for k = 1:nx
    plot3D([0, 1], [X1[idx2[k]], X2_apf[k]], [0, 0], "C3:")
    plot3D(vx*ones(N2), X1[k]*ones(N2), [0, a*pdf.(d1, X1[k])], "C3:")
end
vx = 1
a = 1/pdf(d2, mean(d2))
plot3D(vx*ones(N), y, a*pdf.(d2, y), "C0")    
plot3D(vx*ones(nx), X2_apf, a*pdf.(d2, X2_apf), "C3x")
for k = 1:nx
    plot3D(vx*ones(N2), X2_apf[k]*ones(N2), [0, a*pdf.(d2, X2_apf[k])], "C3:")
end

axis("off")
grid(false)
xlim([-1, 2])
ylim([y0, y1])
"""

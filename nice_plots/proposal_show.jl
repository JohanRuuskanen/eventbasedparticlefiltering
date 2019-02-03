
#using PyCall
using Random
using PyPlot
using StatsBase
using Distributions

#=
PyCall.@pyimport matplotlib2tikz
"""
`savetikz(path; fig = PyPlot.gcf(), extra::Vector{String})`
"""
function savetikz(path; fig = PyPlot.gcf(), extra=[])
    if extra == []
        matplotlib2tikz.save(path,fig, figureheight = "\\figureheight",
                            figurewidth = "\\figurewidth")
    else
        matplotlib2tikz.save(path,fig, figureheight = "\\figureheight",
                             figurewidth = "\\figurewidth",
                             extra_tikzpicture_parameters = PyCall.pybuiltin("set")(extra))
    end
end
=#


N = 1000
N2 = 2
M = 20
V = 2/19* 1/sqrt(2)
nx = 10


x0 = -0.5
x1 = 1.5
y0 = -2
y1 = 2

Random.seed!(1337)

x = ones(N)
y = range(y0, stop=y1, length=1000)

d1 = MixtureModel(map(x -> Normal(x, V), range(-1, stop=1, length=M)))
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

s1 = 8
s2 = 2

fig1 = figure(1)
clf()
vx = 0
a = 1/pdf(d1, mean(d1))
plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0", linewidth=2.0)
for k = 1:nx
    plot3D([0, 1], [X1[idx[k]], X2_bpf[k]], [0, 0], "C1:")
    plot3D(vx*ones(N2), X1[k]*ones(N2), [0, a*pdf.(d1, X1[k])], "C1--")
end
plot3D(vx*ones(nx), X1, a*pdf.(d1, X1), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
vx = 1
a = 1/pdf(d2, mean(d2))
plot3D(vx*ones(N), y, a*pdf.(d2, y), "C0", linewidth=2.0)
for k = 1:nx
    plot3D(vx*ones(N2), X2_bpf[k]*ones(N2), [0, a*pdf.(d2, X2_bpf[k])], "C1--")
end
plot3D(vx*ones(nx), X2_bpf, a*pdf.(d2, X2_bpf), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
axis("off")
#grid("off")
xlim([x0, x1])
ylim([y0, y1])
gca()[:text](0.1, y0-0.35, 0, "t1")
gca()[:text](1.1, y0-0.35, 0, "t2")
#gca()[:w_zaxis][:line][:set_lw](0.0)
#gca()[:set_zticks]([])
#gca()[:w_yaxis][:line][:set_lw](0.0)
#gca()[:set_yticks]([])
gca()[:view_init](30, -35)
draw()

fig2 = figure(2)
clf()
vx = 0
a = 1/pdf(d1, mean(d1))
plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0", linewidth=2.0)
for k = 1:nx
    plot3D([0, 1], [X1[idx[k]], X2_sis[k]], [0, 0], "C2:")
    plot3D(vx*ones(N2), X1[k]*ones(N2), [0, a*pdf.(d1, X1[k])], "C2--")
end
plot3D(vx*ones(nx), X1, a*pdf.(d1, X1), "C2^", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
vx = 1
a = 1/pdf(d2, mean(d2))
plot3D(vx*ones(N), y, a*pdf.(d2, y), "C0", linewidth=2.0)
for k = 1:nx
    plot3D(vx*ones(N2), X2_sis[k]*ones(N2), [0, a*pdf.(d2, X2_sis[k])], "C2--")
end
plot3D(vx*ones(nx), X2_sis, a*pdf.(d2, X2_sis), "C2^", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))

axis("off")
xlim([x0, x1])
ylim([y0, y1])
gca()[:view_init](30, -35)
draw()

fig3 = figure(3)
clf()
vx = 0
a = 1/pdf(d1, mean(d1))
plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0", linewidth=2.0)
for k = 1:nx
    plot3D([0, 1], [X1[idx2[k]], X2_apf[k]], [0, 0], "C3:")
    plot3D(vx*ones(N2), X1[k]*ones(N2), [0, a*pdf.(d1, X1[k])], "C3--")
end
plot3D(vx*ones(nx), X1, a*pdf.(d1, X1), "C3s", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
vx = 1
a = 1/pdf(d2, mean(d2))
plot3D(vx*ones(N), y, a*pdf.(d2, y), "C0", linewidth=2.0)
for k = 1:nx
    plot3D(vx*ones(N2), X2_apf[k]*ones(N2), [0, a*pdf.(d2, X2_apf[k])], "C3--")
end
plot3D(vx*ones(nx), X2_apf, a*pdf.(d2, X2_apf), "C3s", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))

axis("off")
xlim([x0, x1])
ylim([y0, y1])
gca()[:view_init](30, -35)
draw()


"""
Illustrative example for the fallacy of using a simple BPF in an event-based
    setting
"""

using Random
using PyPlot
using StatsBase
using Distributions

# Define parameters
N = 1000
N2 = 2
M = 20
V = 2/19* 1/sqrt(2)
nx = 10

x0 = -0.5
x1 = 1.5
y0 = -2
y1 = 2
y2 = 4

Random.seed!(3)

x = ones(N)
y = range(y0, stop=y1, length=1000)

# Define the distributions
d1 = MixtureModel(map(x -> Normal(x, V), range(-1, stop=1, length=M)))
d2 = MixtureModel(map(x -> Normal(x, V), range(-1, stop=1, length=M)))
d3 = Normal(1, 0.2)

# Propagate particles 3 steps
X1 = rand(d1, nx)
W1 = pdf.(d1, X1)
W1 /= sum(W1)
I1 = rand(Categorical(W1), nx)

X2 = X1[I1] + rand(Normal(0, 0.5), nx)
W2 = pdf.(d2, X2)
W2 /= sum(W2)
I2 = rand(Categorical(W2), nx)

X3 = X2[I2] + rand(Normal(0, 0.5), nx)
W3 = pdf.(d3, X3)
W3 /= sum(W3)
I3 = rand(Categorical(W3), nx)

vx1 = 0
vx2 = 1
vx3 = 2
a1 = 1/pdf(d1, mean(d1))
a2 = 1/pdf(d2, mean(d2))
a3 = 1/pdf(d3, mean(d3))

total_figs = 11
# Step 1
for fig = 1:total_figs
    figure(fig)
    clf()
    plot3D(vx1*ones(N), y, a1*pdf.(d1, y), "C0", linewidth=2.0)
    axis("off")
    xlim([x0, x1])
    ylim([y0, y1])
    gca()[:view_init](30, -35)
end

# step 2
figure(2)
plot3D(vx1*ones(nx), X1, zeros(size(X1)), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))

# Step 3
for fig = 3:4
    figure(fig)
    for k = 1:nx
        plot3D(vx1*ones(N2), X1[k]*ones(N2), [0, a1*pdf.(d1, X1[k])], "C1--",zorder=0)
    end
    plot3D(vx1*ones(nx), X1, a1*pdf.(d1, X1), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
end

# Step 4
for fig = 4:total_figs
    figure(fig)
    plot3D(vx2*ones(N), y, a2*pdf.(d2, y), "C0", linewidth=2.0)
end

# Step 5
for fig = 5:total_figs
    figure(fig)
    plot3D(vx1*ones(nx), X1, a1*pdf.(d1, X1), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1), alpha=0.3)
    for k = 1:nx
        plot3D(vx1*ones(N2), X1[k]*ones(N2), [0, a1*pdf.(d1, X1[k])], "C1--",zorder=0, alpha=0.3)
    end

    plot3D(vx1*ones(nx), X1[I1], a1*pdf.(d1, X1[I1]), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
    for k = 1:nx
        plot3D(vx1*ones(N2), X1[I1[k]]*ones(N2), [0, a1*pdf.(d1, X1[I1[k]])], "C1--",zorder=0)
    end
end

# Step 6
for fig = 6:total_figs
    figure(fig)
    for k = 1:nx
        plot3D([0, 1], [X1[I1[k]], X2[k]], [0, 0], "C1:")
    end
end
figure(6)
plot3D(vx2*ones(nx), X2, zeros(size(X1)), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))

# Step 7
for fig = 7:8
    figure(fig)
    plot3D(vx2*ones(nx), X2, a2*pdf.(d2, X2), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
    for k = 1:nx
        plot3D(vx2*ones(N2), X2[k]*ones(N2), [0, a2*pdf.(d2, X2[k])], "C1--",zorder=0)
    end
end

# Step 8
for fig = 8:total_figs
    figure(fig)
    plot3D(vx3*ones(N), y, a3*pdf.(d3, y), "C0", linewidth=2.0)
end

# Step 9
for fig = 9:total_figs
    figure(fig)
    plot3D(vx2*ones(nx), X2, a2*pdf.(d2, X2), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1), alpha=0.3)
    for k = 1:nx
        plot3D(vx2*ones(N2), X2[k]*ones(N2), [0, a1*pdf.(d2, X2[k])], "C1--",zorder=0, alpha=0.3)
    end

    plot3D(vx2*ones(nx), X2[I2], a2*pdf.(d2, X2[I2]), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
    for k = 1:nx
        plot3D(vx2*ones(N2), X2[I2[k]]*ones(N2), [0, a2*pdf.(d2, X2[I2[k]])], "C1--",zorder=0)
    end
end

# Step 10
for fig = 10:total_figs
    figure(fig)
    for k = 1:nx
        plot3D([1, 2], [X2[I2[k]], X3[k]], [0, 0], "C1:")
    end
end
figure(10)
plot3D(vx3*ones(nx), X3, zeros(size(X3)), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))

# Step 11
figure(11)
plot3D(vx3*ones(nx), X3, a3*pdf.(d3, X3), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
for k = 1:nx
    plot3D(vx3*ones(N2), X3[k]*ones(N2), [0, a3*pdf.(d3, X3[k])], "C1--",zorder=0)
end

#=
figure(2)
clf()
plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0", linewidth=2.0)
plot3D(vx*ones(nx), X1, zeros(size(X1)), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
axis("off")
xlim([x0, x1])
ylim([y0, y1])
gca()[:view_init](30, -35)
draw()

figure(3)
clf()
plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0", linewidth=2.0)
plot3D(vx*ones(nx), X1, a*pdf.(d1, X1), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
for k = 1:nx
    plot3D(vx*ones(N2), X1[k]*ones(N2), [0, a*pdf.(d1, X1[k])], "C1--",zorder=0)
end
axis("off")
xlim([x0, x1])
ylim([y0, y1])
gca()[:view_init](30, -35)
draw()

figure(4)
clf()
# Step 2
plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0", linewidth=2.0)
plot3D(vx*ones(nx), X1, a*pdf.(d1, X1), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
for k = 1:nx
    plot3D(vx*ones(N2), X1[k]*ones(N2), [0, a*pdf.(d1, X1[k])], "C1--",zorder=0)
end
vx = 1
a = 1/pdf(d2, mean(d2))
for k = 1:nx
    plot3D([0, 1], [X1[idx[k]], X2_bpf[k]], [0, 0], "C1:")
end
plot3D(vx*ones(N), y, a*pdf.(d2, y), "C0", linewidth=2.0)
plot3D(vx*ones(nx), X2_bpf, zeros(size(X2_bpf)), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
axis("off")
xlim([x0, x1])
ylim([y0, y1])
gca()[:view_init](30, -35)
draw()


figure(5)
clf()
vx = 0
a = 1/pdf(d1, mean(d1))
plot3D(vx*ones(N), y, a*pdf.(d1, y), "C0", linewidth=2.0)
plot3D(vx*ones(nx), X1, a*pdf.(d1, X1), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
for k = 1:nx
    plot3D(vx*ones(N2), X1[k]*ones(N2), [0, a*pdf.(d1, X1[k])], "C1--",zorder=0)
end
vx = 1
a = 1/pdf(d2, mean(d2))
for k = 1:nx
    plot3D([0, 1], [X1[idx[k]], X2_bpf[k]], [0, 0], "C1:")
end
plot3D(vx*ones(N), y, a*pdf.(d2, y), "C0", linewidth=2.0)
plot3D(vx*ones(nx), X2_bpf, a*pdf.(d2, X2_bpf), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
axis("off")
xlim([x0, x1])
ylim([y0, y1])
gca()[:view_init](30, -35)
draw()


=#
#
#vx = 1
#a = 1/pdf(d2, mean(d2))
#plot3D(vx*ones(N), y, a*pdf.(d2, y), "C0", linewidth=2.0)
#for k = 1:nx
#    plot3D(vx*ones(N2), X2_bpf[k]*ones(N2), [0, a*pdf.(d2, X2_bpf[k])], "C1--")
#end
#plot3D(vx*ones(nx), X2_bpf, a*pdf.(d2, X2_bpf), "C1o", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))


#=
fig2 = figure(2)
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
=#

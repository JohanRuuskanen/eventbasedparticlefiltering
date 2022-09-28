
function plot_measurement_data(pfd::particle_data, y::Array{T,2}; yd=1, nofig=false,
    t_int=[1,1]) where T <: AbstractFloat

    @assert 1 <= yd <= size(pfd.H, 1)

    if t_int == [1,1]
        t_int[2] = size(pfd.X, 3)
    end
    t0 = t_int[1]
    tf = t_int[2]

    Γ = pfd.Γ[t0:tf]

    Y_max = maximum(y[yd, t0:tf])
    Y_min = minimum(y[yd, t0:tf])

    if !nofig
        figure()
    end
    idxs = findall(x -> x == 1, Γ)
    for idx in idxs
        fill_between([idx-1.25, idx-0.75], Y_min-5, Y_max+5,
            facecolor="gray", alpha=0.25, zorder=1)
    end
    plot(y[yd, t0:tf], "r", linewidth=5, alpha=0.5)
    fill_between(collect(0:tf-t0), pfd.H[yd, 1, t0:tf], pfd.H[yd, 2, t0:tf],
        facecolor="green", alpha=0.3, zorder=0, step="mid")


    y_line = PyPlot.matplotlib.lines.Line2D([],[],color="r", linewidth=5,
        alpha=0.5, label=L"$y_k$")
    γk_patch = PyPlot.matplotlib.patches.Patch(color="gray", alpha=0.25,
        label=L"$\gamma_k = 1$")
    Hk_patch = PyPlot.matplotlib.patches.Patch(color="green", alpha=0.5,
        label=L"$H_k$")

    legend(handles=[y_line, γk_patch, Hk_patch])

end

# Plot the particle trace, dont use too large amount of particles!
function plot_particle_trace(pfd::particle_data; x_true=[], xd=1, nofig=false,
    t_int = [1,1])

    @assert 1 <= xd <= size(pfd.X,2)

    if t_int == [1,1]
        t_int[2] = size(pfd.X, 3)
    end
    t0 = t_int[1]
    tf = t_int[2]

    # Transpose out of adjoint
    X = Array{Float64, 2}(pfd.X[:, xd, t0:tf]')
    S = Array{Float64, 2}(pfd.S[:, t0:tf]')
    Γ = pfd.Γ[t0:tf]
    m, n = size(X)

    X_max = maximum(X)
    X_min = minimum(X)

    # Plot
    if !nofig
        figure()
    end
    idxs = findall(x -> x == 1, Γ)
    for idx in idxs
        fill_between([idx-1.25, idx-0.75], X_min-5, X_max+5,
            facecolor="grey", alpha=0.25, zorder=0)
    end
    if !isempty(x_true)
        plot(x_true[xd, t0:tf], "r", linewidth=5, alpha=0.5, zorder=5)
    end
    plot(X, "C0o", markeredgecolor="k", zorder=15)
    for i = 1:m-1
        for j = 1:n
            branch = findall(x-> x == j, S[i+1,:])
            for k in branch
                plot([i-1, i], [X[i, j], X[i+1, k]], "k", zorder=10)
            end
        end
    end
    ylim([X_min-5, X_max+5])

    x_line = PyPlot.matplotlib.lines.Line2D([],[],color="r", linewidth=5,
        alpha=0.5, label=L"$x_k$")
    γk_patch = PyPlot.matplotlib.patches.Patch(color="grey", alpha=0.25,
        label=L"$\gamma_k = 1$")
    Xk_marker = PyPlot.matplotlib.lines.Line2D([],[], color="k", marker="o",
        markerfacecolor="C0", markeredgecolor="k", label=L"$X^i_k$")

    legend(handles=[x_line, γk_patch, Xk_marker])

end

function plot_particle_hist(pfd::particle_data; xd=1, x_true=[], nofig=false,
        colornorm=matplotlib.colors.Normalize(), colormap="YlGnBu", t_int=[1,1])

    @assert 1 <= xd <= size(pfd.X, 2)

    if t_int == [1,1]
        t_int[2] = size(pfd.X, 3)
    end
    t0 = t_int[1]
    tf = t_int[2]

    X = pfd.X[:, xd, t0:tf]
    W = pfd.W[:, t0:tf]
    xh = sum(W.* X, dims=1)
    Γ = pfd.Γ[t0:tf]

    m, n = size(X)

    X_max = maximum(X)
    X_min = minimum(X)

    ylow = X_min - 5
    yhigh = X_max + 5

    y = X[:]
    x = repeat((0:n-1) .- 0.5, inner=m)

    #in average 20 particles per bin
    xe = (0:n) .- 0.5
    ye = range(ylow, stop=yhigh, length=100)

    H = fit(Histogram, (x, y), weights(W[:]), (xe, ye))

    E = H.weights'
    E = (E ./ sum(E, dims=1))

    # Plot
    if !nofig
        figure()
    end
    pcolor(H.edges[1], H.edges[2], E, cmap=get_cmap(colormap), norm=colornorm,
        vmin=0, vmax=0.3)
    idxs = findall(x -> x == 1, Γ)
    for idx in idxs
        fill_between([idx-1.25, idx-0.75], X_min-5, X_max+5,
            facecolor="gray", alpha=0.25, zorder=1)
    end
    plot(xh[:], "r--", linewidth=2, alpha=0.5, zorder=3)
    if !isempty(x_true)
      plot(x_true[xd, t0:tf], "b", linewidth=2, alpha=0.5, zorder=2)
    end
    ylim([ylow, yhigh])

    colorbar()

    x_line = PyPlot.matplotlib.lines.Line2D([], [], color="b", linewidth=2,
        alpha=0.5, label=L"$x_k$")
    xh_line = PyPlot.matplotlib.lines.Line2D([], [], color="r", linewidth=2,
        linestyle="--", alpha=0.5, label=L"$\hat{x}_k$")
    γk_patch = PyPlot.matplotlib.patches.Patch(color="grey", alpha=0.25,
        label=L"$\gamma_k = 1$")

    legend(handles=[x_line, xh_line, γk_patch])

end

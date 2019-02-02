using PyPlot

# Plot the particle trace, dont use too large amount of particles!
function plot_particle_trace(X::Array{Float64,2}, S::Array{Float64,2};
    x_true=[], Γ=[], nofig=false)
    if size(X) != size(S)
        error("Input size must be equal")
    end

    # Transpose out of adjoint
    X = Array{Float64, 2}(X')
    S = Array{Float64, 2}(S')
    m, n = size(X)

    X_max = maximum(X)
    X_min = minimum(X)

    # Plot
    if !nofig
        figure()
    end
    if !isempty(Γ)
        idxs = findall(x -> x == 1, Γ)
        for idx in idxs
            plt[:fill_between]([idx-1.25, idx-0.75], X_min-5, X_max+5,
                facecolor="green", alpha=0.5)
        end
    end
    if !isempty(x_true)
        plot(x_true[:], "r", linewidth=5, alpha=0.5)
    end
    for i = 1:m-1
        for j = 1:n
            branch = findall(x-> x == j, S[i+1,:])
            for k in branch
                plot([i-1, i], [X[i, j], X[i+1, k]], "k")
            end
        end
    end
    plot(X, "C0o", markeredgecolor="k")
    ylim([X_min-5, X_max+5])
end

function plot_data(y::Array{Float64,1}, z::Array{Float64,1}; δ=0, nofig=false)

    if !nofig
        figure()
    end
    plot(y, "r", linewidth=5, alpha=0.5)
    step(z, "C0", linewidth=2, where="post")
    if δ > 0
        plt[:fill_between](collect(0:length(z)-1), z .- δ, z .+ δ,
            facecolor="green", alpha=0.3, zorder=0, step="post")
    end
    legend(["y", "z"])

end

function plot_effective_sample_size(W::Array{Float64, 2}; Γ=[], nofig=false)
    # Transpose out of adjoint
    W = Array{Float64, 2}(W')
    Neff = 1 ./ sum(W.^2, dims=2)
    Neff_max = maximum(Neff)
    Neff_min = 0

    if !nofig
        figure()
    end

    if !isempty(Γ)
        idxs = findall(x -> x == 1, Γ)
        for idx in idxs
            plt[:fill_between]([idx-1.25, idx-0.75], Neff_min, 1.2*Neff_max,
                facecolor="green", alpha=0.5)
        end
    end
    plot(Neff, linewidth=2)
    ylim([Neff_min, 1.2*Neff_max])
end

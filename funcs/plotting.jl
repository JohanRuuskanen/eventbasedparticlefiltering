using PyPlot

# Plot the particle trace, dont use too large amount of particles!
function plot_particle_trace(X::Array{Float64,2}, S::Array{Float64,2};
    x_true=[], Γ=[], fignr=0)
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
    if fignr > 0
        figure(fignr)
    else
        figure()
    end
    clf()
    if !isempty(Γ)
        idxs = findall(x -> x == 1, Γ)
        for idx in idxs
            plt[:fill_between]([idx-1.25, idx-0.75], X_min-5, X_max+5, facecolor="green", alpha=0.3)
        end
    end

    if !isempty(x_true)
        plot(x_true[:], "r", linewidth=5, alpha=0.5)
    end
    for i = 1:m-1
        for j = 1:n
            branch = findall(x-> x == j, S[i+1,:])
            for k in branch
                plot([i-1, i], [X[i, j], X[i+1, k]], "k--")
            end
        end
    end
    plot(X, "bo", markeredgecolor="k")
    ylim([X_min-5, X_max+5])


    # Add plotting of trace from end?

end


function multinomial_resampling(W::Array{Float64, 1})
    N = size(W, 1)
    idx = rand(Categorical(W), N)
    return idx
end


function systematic_resampling(W::Array{Float64, 1})
    N = size(W, 1)
    idx = collect(1:N)
    wc = cumsum(W)
    u = ((collect(0:(N-1)) .+ rand()) / N)
    c = 1
    for i = 1:N
        while wc[c] < u[i]
            c = c + 1
        end
        idx[i] = c
    end
    return idx
end

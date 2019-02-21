@testset "test_resampling" begin

    N = 1000
    M = 1000

    W = ones(N, M)
    for k = 1:N
        W[k, :] /= N
    end

    Y = ones(N) * N
    X = zeros(N)

    idx = systematic_resampling(W[1, :])
    @test size(idx, 1) == N
    @test typeof(idx) == Array{Int64, 1}

    for k = 1:M
        idx = systematic_resampling(W[k, :])
        for i in idx; X[i] += 1 end
    end
    diff = X - Y

    # Less than 5% off for each value
    @test (diff .< N * 0.05) == BitArray{1}(ones(N))



end

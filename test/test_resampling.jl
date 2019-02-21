@testset "test_resampling" begin

    function sampling_uniform(N, M, res_func::Function)
        Y = ones(N) * M
        X = zeros(N)

        W = ones(N, M) ./ N

        for k = 1:M
            idx = res_func(W[:, k])
            for i in idx; X[i] += 1 end
        end
        diff = abs.(X - Y)

        return diff
    end

    Random.seed!(2)

    N = 1000
    M = 10000

    W = ones(N, 2) ./ N

    idx = EP.resampling_systematic(W[:, 1])
    @test size(idx, 1) == N
    @test typeof(idx) == Array{Int64, 1}

    idx = EP.resampling_multinomial(W[:, 1])
    @test size(idx, 1) == N
    @test typeof(idx) == Array{Int64, 1}

    diff_systematic = sampling_uniform(N, M, EP.resampling_systematic)
    diff_multinomial = sampling_uniform(N, M, EP.resampling_multinomial)

    # Less than 15% off for each value
    @test (diff_systematic .< M * 0.15) == BitArray{1}(ones(N))
    @test (diff_multinomial .< M * 0.15) == BitArray{1}(ones(N))


end

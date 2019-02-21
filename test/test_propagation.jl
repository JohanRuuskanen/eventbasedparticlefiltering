@testset "test_propagation" begin

    Random.seed!(123)

    m = 5; n = 4; T = 1000; N = 1000
    sys = create_params(m, n, T)

    X = zeros(N, m, T)
    X_new = propagation_bootstrap(X[:, :, 1], sys)

    err_mean = abs.(mean(X_new, dims=1))
    err_var = abs.(var(X_new, dims=1)) .- 1

    @test size(X_new) == size(X[:,:,1])
    @test typeof(X_new) == Array{Float64, 2}
    @test any(x-> x == true, err_mean .< 0.15)
    @test any(x-> x == true, err_var .< 0.15)

end

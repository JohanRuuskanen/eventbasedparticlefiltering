@testset "test_propagation" begin

    Random.seed!(123)

    m = 5; n = 4; T = 1000; N = 1000
    sys = create_params(m, n, T)

    X = zeros(N, m, T)
    propagation_bootstrap!(view(X, :, :, 2), X[:, :, 1], sys)

    X_new = X[:, :, 2]

    err_mean = abs.(mean(X_new, dims=1))
    err_var = abs.(var(X_new, dims=1)) .- 1

    @test size(X_new) == size(X[:,:,1])
    @test typeof(X_new) == Array{Float64, 2}
    @test any(x-> x == true, err_mean .< 0.15)
    @test any(x-> x == true, err_var .< 0.15)

    N = 200
    M = 10
    T = 1000
    δ = 4

    A = reshape([1.0], 1, 1)
    C = reshape([1.0], 1, 1)

    nx = size(A, 1)
    ny = size(C, 1)

    Q = 1*Matrix{Float64}(I, nx, nx)
    R = 0.1*Matrix{Float64}(I, ny, ny)

    sys = sys_params(A, C, Q, R, T)
    x, y = sim_sys(sys)

    y = [0.5]; z = [1.0]; δ = 2
    Vn = 0.1; γ = 0
    yh = vcat(range(z .- δ, stop=(z .+ δ), length=M)...)
    Xr = randn(N, 1)

    X, q_list = propagation_locallyOptimal(Xr, z, sys, yh, Vn, γ)

    @test size(q_list) == (N,)
    @test size(X) == (N, nx)
    @test mean(X) > mean(Xr)

end

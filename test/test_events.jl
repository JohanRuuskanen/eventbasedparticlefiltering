@testset "test_events" begin

    Random.seed!(123)

    T = 1000; N = 200; M = 10
    nx = 1; ny = 1

    A = reshape([0.9], (nx, nx))
    C = reshape([1.0], (ny, nx))
    Q = 1*Matrix{Float64}(I, nx, nx)
    R = 0.1*Matrix{Float64}(I, ny, ny)

    sys = sys_params(A, C, Q, R, T)
    x, y = sim_sys(sys)

    par_SOD = pf_params(N, "SOD", 0)
    par_MBT = pf_params(N, "MBT", 0)

    k = 5
    δ = 2

    z1 = zeros(Float64, 1); z0 = zeros(Float64, 1); yh = zeros(Float64, M)
    γ = eventSampling!(z1, yh, y[:, k], z0, x[:, k-1], sys, par_SOD, 0, M)
    @test typeof(z1) == Array{Float64,1}
    @test size(z1) == size(y[:, k])
    @test typeof(yh) == Array{Float64,1}
    @test size(yh) == (M,)

    z1 = zeros(Float64, 1); z0 = zeros(Float64, 1); yh = zeros(Float64, M)
    γ = eventSampling!(z1, yh, y[:, k], z0, x[:, k-1], sys, par_MBT, 0, M)
    @test typeof(z1) == Array{Float64,1}
    @test size(z1) == size(y[:, k])
    @test typeof(yh) == Array{Float64,1}
    @test size(yh) == (M,)

    z1 = zeros(Float64, 1); z0 = zeros(Float64, 1); yh = zeros(Float64, M)
    γ = eventSampling!(z1, yh, y[:, k], z0, x[:, k-1], sys, par_SOD, 1e6, M)
    @test typeof(z1) == Array{Float64, 1}
    @test size(z1) == size(y[:, k])
    @test z1 == z0
    @test γ == 0
    @test size(yh) == (M,)

    z1 = zeros(Float64, 1); z0 = zeros(Float64, 1); yh = zeros(Float64, M)
    γ = eventSampling!(z1, yh, y[:, k], z0, x[:, k-1], sys, par_MBT, 1e6, M)
    @test typeof(z1) == Array{Float64, 1}
    @test size(z1) == size(y[:, k])
    @test z1 == sys.C * sys.A * x[:, k-1]
    @test γ == 0
    @test size(yh) == (M,)

    # Test SOD
    Z = zeros(Float64, size(y))
    Γ = zeros(Float64, T)
    Yh = zeros(Float64, size(y, 1), T, M)
    for k = 2:T
        Γ[k] =  eventSampling!(view(Z, :, k), view(Yh, 1, k, :), y[:, k],
            Z[:, k-1], x[:, k-1], sys, par_SOD, δ, 10)
    end
    idx_0 = findall(x -> x == 0, Γ)
    idx_1 = findall(x -> x == 1, Γ)

    @test Z[:, idx_1] == y[:, idx_1]
    @test all(x -> isnan(x), Yh[:, idx_1, :])
    @test all(x -> x == false, [y[:, idx_0[i]] == Z[:, idx_0[i]]
        for i in 2:length(idx_0)])
    @test minimum(std(Yh[:, idx_0[2:end], :], dims=3)) >= sqrt((2*δ)^2 / 12)
    @test maximum(Yh[:, idx_0[2:end], :], dims=3) == Z[:, idx_0[2:end], :] .+ δ
    @test minimum(Yh[:, idx_0[2:end], :], dims=3) == Z[:, idx_0[2:end], :] .- δ

    # Test MBT
    Z = zeros(Float64, size(y))
    Γ = zeros(Float64, T)
    Yh = zeros(Float64, size(y, 1), T, M)
    for k = 2:T
        Γ[k] =  eventSampling!(view(Z, :, k), view(Yh, 1, k, :), y[:, k],
            Z[:, k-1], x[:, k-1], sys, par_MBT, δ, 10)
    end
    idx_0 = findall(x -> x == 0, Γ)
    idx_1 = findall(x -> x == 1, Γ)

    @test Z[:, idx_1] == y[:, idx_1]
    @test all(x -> isnan(x), Yh[:, idx_1, :])
    @test all(x -> x == false, [y[:, idx_0[i]] == Z[:, idx_0[i]]
        for i in 2:length(idx_0)])
    @test minimum(std(Yh[:, idx_0[2:end], :], dims=3)) >= sqrt((2*δ)^2 / 12)
    @test maximum(Yh[:, idx_0[2:end], :], dims=3) == Z[:, idx_0[2:end], :] .+ δ
    @test minimum(Yh[:, idx_0[2:end], :], dims=3) == Z[:, idx_0[2:end], :] .- δ


end

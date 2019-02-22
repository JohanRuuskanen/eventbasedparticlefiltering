@testset "test_weighting" begin

    nx = 10; ny = 1
    N = 1000; M = 20
    δ = 4

    A = Matrix{Float64}(I, nx, nx)
    C = ones(ny, nx)

    Q = 1*Matrix{Float64}(I, nx, nx)
    R = 0.1*Matrix{Float64}(I, ny, ny)

    sys = sys_params(A, C, Q, R, T)
    x, y = sim_sys(sys)

    par = pf_params(N, "SOD", N/2)

    W = zeros(N)
    X = randn(N, nx)
    z = reshape([1.0], 1)
    Wr = ones(N) ./ N
    Xr = randn(N, nx)
    yh = vcat(range(z .- δ, stop=(z .+ δ), length=M)...)

    q_list = Array{Distribution}(undef, N)
    q_aux_list = Array{Distribution}(undef, N)
    for k = 1:N
        q_list[k] = MvNormal(zeros(nx), Matrix{Float64}(I, nx, nx))
        q_aux_list[k] = MvNormal(zeros(ny), Matrix{Float64}(I, ny, ny))
    end

    r = 0; γ = 0
    calculate_weights!(W, X, z, Wr, Xr, yh, q_list, q_aux_list, r, γ, δ, sys, par)
    @test size(W) == (N,)
    @test typeof(W) == Array{Float64,1}

    r = 0; γ = 1
    calculate_weights!(W, X, z, Wr, Xr, yh, q_list, q_aux_list, r, γ, δ, sys, par)
    @test size(W) == (N,)
    @test typeof(W) == Array{Float64,1}

    r = 1; γ = 0
    calculate_weights!(W, X, z, Wr, Xr, yh, q_list, q_aux_list, r, γ, δ, sys, par)
    @test size(W) == (N,)
    @test typeof(W) == Array{Float64,1}

    r = 1; γ = 1
    calculate_weights!(W, X, z, Wr, Xr, yh, q_list, q_aux_list, r, γ, δ, sys, par)
    @test size(W) == (N,)
    @test typeof(W) == Array{Float64,1}

end

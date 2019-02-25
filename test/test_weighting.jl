@testset "test_weighting" begin

    function local_tests(r, γ)

        println("With r:$(r), γ:$(γ)")

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


        calculate_weights!(W, X, z, Wr, Xr, yh, q_list, q_aux_list, r, γ, δ, sys, par)

        # Check that the calculated weights have the correct size and type
        @test size(W) == (N,)
        @test typeof(W) == Array{Float64,1}
    end

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

    # Test the weight calculation functions with/without resampling and
    # with/without triggered event
    r = 0; γ = 0
    local_tests(r, γ)

    r = 0; γ = 1
    local_tests(r, γ)

    r = 1; γ = 0
    local_tests(r, γ)

    r = 1; γ = 1
    local_tests(r, γ)

end

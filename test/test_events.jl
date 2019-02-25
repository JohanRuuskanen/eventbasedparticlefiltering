@testset "test_events" begin

    function local_tests(par)
        println("Testing "*par.eventKernel)

        Z = zeros(Float64, size(y))
        Γ = zeros(Float64, T)
        Yh = zeros(Float64, size(y, 1), T, M)
        for k = 2:T
            Γ[k] =  eventSampling!(view(Z, :, k), view(Yh, 1, k, :), y[:, k],
                Z[:, k-1], x[:, k-1], sys, par, δ, M)
        end
        idx_0 = findall(x -> x == 0, Γ) # Find the non-triggered time instances
        idx_1 = findall(x -> x == 1, Γ) # Find all triggered events

        # Logged measurements should be the same as the true measurements
        # when triggered
        @test Z[:, idx_1] == y[:, idx_1]

        # Returned kernel discretization should be NaN when an event is generated
        @test all(x -> isnan(x), Yh[:, idx_1, :])

        # Logged measurements should not be the same as the true measurements when
        # not triggered. This "can" happen but is very unlikely when dealing with
        # double precision
        @test all(x -> x == false, [y[:, idx_0[i]] == Z[:, idx_0[i]]
            for i in 2:length(idx_0)])

        # The maximum and minimum discretization of the event kernel should occur on
        # the boundaries of Z ± δ
        @test maximum(Yh[:, idx_0[2:end], :], dims=3) == Z[:, idx_0[2:end], :] .+ δ
        @test minimum(Yh[:, idx_0[2:end], :], dims=3) == Z[:, idx_0[2:end], :] .- δ
    end


    # Initialize random seed and a testing system
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

    k = 5; δ = 2

    # Test SOD sampling on the simulated system
    local_tests(par_SOD)

    # Test MBT sampling on the simulated system
    local_tests(par_MBT)

end

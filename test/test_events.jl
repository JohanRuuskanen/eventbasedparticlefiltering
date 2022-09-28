@testset "test_events" begin


    # Initialize random seed and a testing system
    Random.seed!(123)

    ####
    # Test the trigger
    ####

    # Test for the bootstrap filter
    ny = 5
    nx = 2
    N = 3
    T = 10
    δ = collect(1.0:ny)

    y = zeros(Float32, ny, T)
    y[1, 2] = 0.5
    y[2, 3] = 1; y[3, 3] = 2;
    y[1, 4] = 0.5; y[5, 4] = 4;
    y[1, 5] = 0.99; y[2, 5] = -1.99; y[3, 5] = 2.99; y[4, 5] = -3.99; y[5, 5] = 4.99
    y[1, 6] = 1.5
    y[2, 7] = -1.5; y[3, 7] = 3.5
    y[4, 8] = -10
    y[2, 9] = -5; y[4, 9] = 5
    y[5, 10] = 100000000

    sys = system(x -> 0, y -> 0, nx, ny, T)
    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(δ), pftype=pftype_bootstrap())
    pfd = EP.generate_pfd(opt, T=Float32)

    for k = 1:T
        pfd.H[:, :, k] = hcat(zeros(ny)-δ, zeros(ny)+δ)
    end

    Γ_true = [false, false, false, false, false, true, true, true, true, true]
    for k = 1:T
        EP.eventtrigger!(pfd, y, k, opt)
    end
    @test all(pfd.Γ .== Γ_true)

    # Test for the auxiliary filter, only works for ny = 1 so far!
    D = 5
    ny = 1
    δ = [1.0]

    y = zeros(ny, T)
    y[1, :] = range(0.0, stop=2.0, length=T)

    sys = system(x -> 0, y -> 0, nx, ny, T)
    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(δ),
        pftype=pftype_auxiliary(qv -> 0, q -> 0, D))
    pfd = EP.generate_pfd(opt, T=Float32)

    Γ_true = [false, false, false, false, false, true, false, false, false, false]
    μh_test = zeros(Bool, D, T)
    Vh_test = zeros(Bool, D, T)
    for k = 2:T
        EP.generate_Hk!(pfd, y, k, opt)
        EP.eventtrigger!(pfd, y, k, opt)
        for j = 1:D
            μh = mean(pfd.Hh[j, 1])
            Vh = cov(pfd.Hh[j, 1])

            μh_test[j, k] = all(pfd.H[:, 1, k] .<= μh .<= pfd.H[:, 2, k])
        end
    end
    @test all(pfd.Γ .== Γ_true)
    @test all([p <: MvNormal for p in typeof.(pfd.Hh[:, 1])])
    @test all(μh_test[:, 2:end])


    ####
    # Test the different kernels
    ####

    T = 2
    N = 10000
    MC = 10000

    sys, _, _, _, _, p0 = EP.generate_example_system(T, type="nonlinear_ny2")
    δ = collect(1.0:sys.ny)

    k = 2

    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(δ),
        pftype=pftype_auxiliary(qv -> 0, q -> 0, 5))

    pfd = EP.generate_pfd(opt, T=Float32)

    #pfd.Z[:, 1] .= collect(1.0:sys.ny)
    y = zeros(Float32, sys.ny, T)
    y[:, 1] .= collect(1.0:sys.ny)

    pfd.X[:, :, 1] = rand(p0, N)'
    yh_mc, _ = estimate_output_monte_carlo(sys, p0)

    # Test SOD
    kernel = kernel_SOD(δ)
    opt = ebpf_options(sys=sys, N=N, kernel=kernel, pftype=pftype_bootstrap())
    EP.generate_Hk!(pfd, y, k, opt)
    @test pfd.H[:, 1, k] == zeros(sys.ny)
    @test pfd.H[:, 2, k] == collect(2:2:2*sys.ny)
    @test pfd.H[:, 1, k] + (pfd.H[:, 2, k] - pfd.H[:, 1, k]) / 2 == y[:, 1]

    # Test IBT
    kernel = kernel_IBT(δ)
    opt = ebpf_options(sys=sys, N=N, kernel=kernel, pftype=pftype_bootstrap())
    EP.generate_Hk!(pfd, y, k, opt)
    yh = pfd.H[:, 1, k] + (pfd.H[:, 2, k] - pfd.H[:, 1, k])/2
    @test isapprox(pfd.H[:, 1, k], yh .- δ)
    @test isapprox(pfd.H[:, 2, k], yh .+ δ)
    @test norm(2*(yh - yh_mc) ./ (yh + yh_mc)) < 0.05

end

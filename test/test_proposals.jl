@testset "test_proposals" begin

    # Test the linearization
    T = 100
    N = 100
    sys, A, C, Q, R, p0 = EP.generate_example_system(T, type="nonlinear_ny2")
    q, qv = EP.genprop_linear_gaussian_noise()
    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD([0.0]),
        pftype=pftype_auxiliary(qv, q, 5))

    @test try q([100.0, 0.0, 100.0, 0.0], 1, [140.0, 1.0], opt); true catch; false end
    @test try q([100.0, 0.0, 100.0, 0.0], 1, [140.0, 1.0], opt, V = zeros(2,2)); true catch; false end
    @test try qv([100.0, 0.0, 100.0, 0.0], 1, opt); true catch; false end

    v1 = q([100.0, 0.0, 100.0, 0.0], 1, [140.0, 1.0], opt)
    v2 = q([100.0, 0.0, 100.0, 0.0], 1, [140.0, 1.0], opt, V = zeros(2,2))
    v3 = qv([100.0, 0.0, 100.0, 0.0], 1, opt)

    @test all(x -> isfinite(x), hcat(mean(v1), cov(v1)))
    @test all(x -> isfinite(x), hcat(mean(v2), cov(v2)))
    @test all(x -> isfinite(x), hcat(mean(v3), cov(v3)))

    # Test classic nonlinear
    T = 100
    N = 100
    sys, A, C, Q, R, p0 = EP.generate_example_system(T, type="nonlinear_classic")
    q, qv = EP.genprop_EMM_for_nonlinearclassic()
    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD([0.0]),
        pftype=pftype_auxiliary(qv, q, 5))

    @test try q([0.0], 1, [1.0], opt); true catch; false end
    @test try q([0.0], 1, [1.0], opt, V = zeros(1,1)); true catch; false end
    @test try qv([0.0], 1, opt); true catch; false end

    v1 = q([0.0], 1, [1.0], opt)
    v2 = q([0.0], 1, [1.0], opt, V = zeros(1,1))
    v3 = qv([0.0], 1, opt)

    @test all(x -> isfinite(x), hcat(mean(v1), cov(v1)))
    @test all(x -> isfinite(x), hcat(mean(v2), cov(v2)))
    @test all(x -> isfinite(x), hcat(mean(v3), cov(v3)))


end

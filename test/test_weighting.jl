@testset "test_weighting" begin

    function take_step_kalman(y, sys_type)
        sys, A, C, Q, R, p0 = EP.generate_example_system(T, type=sys_type)

        S1 = C*Q*C' + R
        K1 = Q*C' / S1
        μ1_1 = mean(p0) + K1*(y[:, 1] - C*mean(p0))
        Σ1_1 = (I - K1*C)*Q

        μ2_1 = A*μ1_1
        Σ2_1 = A*Σ1_1*A' + Q
        S2 = C*Σ2_1*C' + R
        K2 = Σ2_1*C' / S2
        μ2_2 = μ2_1 + K2*(y[:, 2] - C*μ2_1)
        Σ2_2 = (I - K2*C)*Σ2_1

        return μ2_2, Σ2_2
    end

    function take_step!(pfd, y, opt)
        EP.create_qv_list!(pfd, y, 2, opt)

        pfd.V[:, 1] .= pfd.W[:, 1] .* sum(pfd.qv_list[:, :, 1], dims=1)[:]
        idx = EP.resampling_systematic(pfd.V[:, 1])

        pfd.Xr[:, :, 1] = pfd.X[idx, :, 1]
        pfd.qv_list[:, :, 1] = pfd.qv_list[:, idx, 1]
        pfd.S[:, 2] .= idx

        EP.create_q_list!(pfd, y, 2, opt)

        for i = 1:opt.N
            pfd.X[i, :, 2] .= rand(pfd.q_list[i, 1])
        end

        EP.calculate_weights!(pfd, y, 2, opt)
    end

    q, qv = EP.genprop_linear_gaussian_noise()

    Random.seed!(123)

    T = 1;
    N = 10000
    sys, A, C, Q, R, p0 = EP.generate_example_system(T, type="linear_ny2")
    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_bootstrap())

    ####
    # Test weight normalization
    ####

    pfd = EP.generate_pfd(opt, T=Float32)

    pfd.W = rand(Float32, N, T);
    tmp = sum(pfd.W[:, 1]);
    pfd.W[123,1] = 0;
    W_orig = copy(pfd.W)
    pfd.W = log.(pfd.W)

    EP.normalize_weights!(view(pfd.W, :, 1), pfd, 1, opt)

    @test pfd.fail[1] == false
    @test typeof(pfd.W) == Array{Float32, 2}
    @test all(k -> isfinite(k), pfd.W[:, 1])
    @test isapprox(sum(pfd.W[:, 1]), 1.0)
    @test isapprox(pfd.W[:, 1]*tmp, W_orig)

    pfd.W = log.(zeros(Float32, N, T))

    EP.normalize_weights!(view(pfd.W, :, 1), pfd, 1, opt)

    @test pfd.fail[1] == true
    @test typeof(pfd.W) == Array{Float32, 2}
    @test all(k -> isfinite(k), pfd.W[:, 1])
    @test isapprox(sum(pfd.W[:, 1]), 1.0)
    @test pfd.W[:, 1] == ones(Float32, N) ./ N

    ####
    # Test weighting & triggering prob calculation
    ####

    T = 2;
    N = 100000
    sys_type = "linear_ny2"
    sys, A, C, Q, R, p0 = EP.generate_example_system(T, type=sys_type)

    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_bootstrap())

    x, y = sim_sys(sys, x0=rand(p0))
    # Bootstrap trigger instance
    μ_true, Σ_true = take_step_kalman(y, sys_type)

    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_bootstrap())
    pfd = EP.generate_pfd(opt, T=Float32)


    pfd.X[:, :, 1] .= rand(p0, N)'
    for i = 1:N
        pfd.W[i, 1] = pdf(sys.py(pfd.X[i, :, 1], 1), y[:, 1])
    end
    pfd.W[:, 1] ./= sum(pfd.W[:, 1])

    pfd.Γ[1] = 1
    pfd.Γ[2] = 1

    take_step!(pfd, y, opt)

    μh = mean(pfd.X[:, :, 2], Weights(pfd.W[:, 2]), dims=1)[:]
    Σh = cov(pfd.X[:, :, 2], Weights(pfd.W[:, 2]), corrected=false)

    @test norm(μh, Inf) > 0.1
    @test norm(Σh, Inf) > 0.1

    @test norm(μ_true - μh, Inf) < 0.15
    @test norm(Σ_true - Σh, Inf) < 0.15

    @test all(x -> isfinite(x), pfd.p_trig)
    @test all(0 .<= pfd.p_trig .<= 1)

    # bootstrap non-trigger instance 1-D
    N = 100
    sys, _, _, _, _, p0 = EP.generate_example_system(T, type="linear_ny1")
    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_bootstrap())
    pfd = EP.generate_pfd(opt, T=Float32)

    pfd.X[:, :, 1] = rand(p0, N)'

    pfd.H[1, 1, 1] = 0.0
    pfd.H[1, 2, 1] = 1.0

    pfd.Γ[1] = 0

    EP.calculate_weights!(pfd, y, 1, opt)

    @test all(k -> isfinite(k), pfd.W[:, 1])
    @test all(pfd.W[:, 1] .>= 0)
    @test typeof(pfd.W) == Array{Float32, 2}
    @test isapprox(sum(pfd.W[:, 1]), 1.0)

    @test all(x -> isfinite(x), pfd.p_trig)
    @test all(0 .<= pfd.p_trig .<= 1)

    # bootstrap non-trigger instance N-d
    N = 100
    sys, _, _, _, _, p0 = EP.generate_example_system(T, type="linear_ny2")
    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_bootstrap(),
        likelihood=likelihood_MC(M=1))
    pfd = EP.generate_pfd(opt, T=Float32)

    pfd.X[:, :, 1] = rand(p0, N)'

    pfd.H[:, 1, 1] = zeros(sys.ny)
    pfd.H[:, 2, 1] = collect(1.0:sys.ny)

    pfd.Γ[1] = 0

    EP.calculate_weights!(pfd, y, 1, opt)

    @test all(k -> isfinite(k), pfd.W[:, 1])
    @test all(pfd.W[:, 1] .>= 0)
    @test typeof(pfd.W) == Array{Float32, 2}
    @test isapprox(sum(pfd.W[:, 1]), 1.0)

    @test all(x -> isfinite(x), pfd.p_trig)
    @test all(0 .<= pfd.p_trig .<= 1)

    # BPF / APF 2 step check
    # only 1-D as multidim approximation of the uniform not implemented yet

    T = 2
    N = 100000
    sys, A, C, Q, R, p0 = EP.generate_example_system(T, type="linear_ny1")
    opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_auxiliary(qv, q, 5))
    pfd = EP.generate_pfd(opt, T=Float32)

    x, y = sim_sys(sys, x0=rand(p0), T=Float32)

    μ_true, Σ_true = take_step_kalman(y, "linear_ny1")

    opt1 = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_bootstrap())
    opt2 = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_auxiliary(qv, q, 5))

    pfd1 = EP.generate_pfd(opt1, T=Float32)
    pfd2 = EP.generate_pfd(opt2, T=Float32)

    pfd1.X[:, :, 1] .= rand(p0, N)'
    for i = 1:N
        pfd1.W[i, 1] = pdf(sys.py(pfd1.X[i, :, 1], 1), y[:, 1])
    end
    pfd1.W[:, 1] ./= sum(pfd1.W[:, 1])

    pfd2.X[:, :, 1] .= rand(p0, N)'
    for i = 1:N
        pfd2.W[i, 1] = pdf(sys.py(pfd2.X[i, :, 1], 1), y[:, 1])
    end
    pfd2.W[:, 1] ./= sum(pfd2.W[:, 1])

    pfd1.Γ[1] = 1
    pfd1.Γ[2] = 1
    pfd2.Γ[1] = 1
    pfd2.Γ[2] = 1

    take_step!(pfd1, y, opt1)
    take_step!(pfd2, y, opt2)

    μh_bpf = mean(pfd1.X[:, :, 2], Weights(pfd1.W[:, 2]), dims=1)[:]
    Σh_bpf = cov(pfd1.X[:, :, 2], Weights(pfd1.W[:, 2]), corrected=false)

    μh_apf = mean(pfd2.X[:, :, 2], Weights(pfd2.W[:, 2]), dims=1)[:]
    Σh_apf = cov(pfd2.X[:, :, 2], Weights(pfd2.W[:, 2]), corrected=false)

    @test norm(μ_true - μh_bpf, Inf) < 0.1
    @test norm(Σ_true - Σh_bpf, Inf) < 0.1
    @test norm(μ_true - μh_apf, Inf) < 0.1
    @test norm(Σ_true - Σh_apf, Inf) < 0.1

    @test all(x -> isfinite(x), pfd.p_trig)
    @test all(0 .<= pfd.p_trig .<= 1)

    # non-trigger step
    T = 2
    N = 10000

    sys, A, C, Q, R, p0 = EP.generate_example_system(T, type="linear_ny1")

    opt1 = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_bootstrap())
    opt2 = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(ones(sys.ny)),
        pftype=pftype_auxiliary(qv, q, 5))

    pfd1 = EP.generate_pfd(opt1)
    pfd2 = EP.generate_pfd(opt2)

    pfd1.X[:, :, 1] = rand(p0, N)'
    pfd1.W[:, 1] = ones(N) ./ N
    pfd2.X[:, :, 1] = rand(p0, N)'
    pfd2.W[:, 1] = ones(N) ./ N

    pfd1.H[1, 1, 2] = 0.0
    pfd1.H[1, 2, 2] = 1.0
    pfd2.H[1, 1, 2] = 0.0
    pfd2.H[1, 2, 2] = 1.0

    pfd1.Γ[1:2] .= 0
    pfd2.Γ[1:2] .= 0

    μh = collect(range(pfd2.H[1, 1, 2], stop=pfd2.H[1, 2, 2], length=opt2.pftype.D))
    Vh = (pfd2.H[1, 2, 2] - pfd2.H[1, 1, 2])/opt2.pftype.D * 0.5
    for j = 1:opt2.pftype.D
        pfd2.Hh[j, 1] =  MvNormal([μh[j]], Vh)
    end

    take_step!(pfd1, y, opt1)
    take_step!(pfd2, y, opt2)

    μh_bpf = mean(pfd1.X[:, :, 2], Weights(pfd1.W[:, 2]), dims=1)[:]
    Σh_bpf = cov(pfd1.X[:, :, 2], Weights(pfd1.W[:, 2]), corrected=false)

    μh_apf = mean(pfd2.X[:, :, 2], Weights(pfd2.W[:, 2]), dims=1)[:]
    Σh_apf = cov(pfd2.X[:, :, 2], Weights(pfd2.W[:, 2]), corrected=false)

    @test norm(μh_apf- μh_bpf, Inf) < 0.1
    @test norm(Σh_apf - Σh_bpf, Inf) < 0.1

    @test all(k -> isfinite(k), pfd1.W[:, 2])
    @test all(pfd1.W[:, 2] .>= 0)
    @test typeof(pfd1.W) == Array{Float64, 2}
    @test isapprox(sum(pfd1.W[:, 2]), 1.0)

    @test all(x -> isfinite(x), pfd1.p_trig)
    @test all(0 .<= pfd1.p_trig .<= 1)

    @test all(k -> isfinite(k), pfd2.W[:, 2])
    @test all(pfd2.W[:, 2] .>= 0)
    @test typeof(pfd2.W) == Array{Float64, 2}
    @test isapprox(sum(pfd2.W[:, 2]), 1.0)

    @test all(x -> isfinite(x), pfd2.p_trig)
    @test all(0 .<= pfd2.p_trig .<= 1)


end

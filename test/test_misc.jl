@testset "test_misc" begin

    Random.seed!(123)

    # Parameters
    N = 100
    T = 200


    ####
    # Simply test if things are executable and does not crash
    ####

    sys1, A1, C1, Q1, R1, p0_1 = EP.generate_example_system(T, type="linear_ny1")
    sys2, A2, C2, Q2, R2, p0_2 = EP.generate_example_system(T, type="linear_ny2")
    sys3, A3, C3, Q3, R3, p0_3 = EP.generate_example_system(T, type="nonlinear_ny2")
    sys4, A4, C4, Q4, R4, p0_4 = EP.generate_example_system(T, type="nonlinear_SOD_pathological")
    sys5, A5, C5, Q5, R5, p0_5 = EP.generate_example_system(T, type="nonlinear_classic")

    @test try kernel_SOD([1.0, 2.0]); true catch; false end
    @test try kernel_IBT([1.0, 2.0]); true catch; false end

    @test try pftype_bootstrap(); true catch; false end
    @test try pftype_auxiliary(x -> 0, y -> 0, 2); true catch; false end

    @test try ebpf_options(sys=sys1, N=1000, kernel=kernel_SOD([1.0]),
            pftype=pftype_bootstrap()); true catch; false end
    @test try ebpf_options(sys=sys4, N=2000, kernel=kernel_IBT([1.0, 2.0, 3.0]),
            pftype=pftype_auxiliary(x -> 0, y -> 0, 2)); true catch; false end

    opt = ebpf_options(sys=sys4, N=2000, kernel=kernel_IBT([1.0, 2.0, 3.0]),
            pftype=pftype_auxiliary(x -> 0, y -> 0, 2))

    @test try EP.particle_data( X = ones(Float32, 3, 2, 10),
                                Xr = ones(Float32, 3, 2, 10),
                                Xp = ones(Float32, 3, 2, 10),
                                W = ones(Float32, 3, 10),
                                V = ones(Float32, 3, 10),
                                H = ones(Float32, 2, 2, 10),
                                Hh = Array{Distribution, 2}(undef, 5, 10),
                                qv_list = ones(Float32, 2, 3, 10),
                                q_list = Array{Distribution, 2}(undef, 3, 10),
                                S = ones(Int64, 3, 10),
                                Γ = ones(Bool, 10),
                                fail = ones(Bool, 10),
                                p_trig = zeros(Float32, 10)); true catch; false end

    @test try EP.generate_pfd(opt, T=Float16); true catch; false end

    @test try EP.err_result(rand(Float32, 10),
                            rand(Float32, 10),
                            rand(Float32, 10),
                            rand(Float32),
                            rand(Float32),
                            rand(Float32),
                            rand(Int64),
                            rand(Int64),
                            rand(Float32),
                            rand(Float32, 10),
                            rand(Int64)); true catch; false end

    @test try sim_sys(sys1); true catch; false end
    @test try sim_sys(sys1, x0=rand(sys1.nx)); true catch; false end

    @test try EP.fix_sym(ones(10, 10)); true catch; false end

    ####
    # Perform function specific tests
    ####

    function test_pred(sys, p0)
        # Empirical distribution

        N = 1000

        opt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT(zeros(sys.ny)),
                            pftype=pftype_bootstrap())

        pfd = EP.generate_pfd(opt)

        pfd.X[:, :, 1] = rand(p0, N)'
        pfd.W[:, 1] .= 1/N

        @test sum(pfd.Xp[:, :, 1]) == 0

        Y = EP.estimate_py_pred!(pfd, 2, opt)

        @test sum(pfd.Xp[:, :, 1]) > 0
        @test length(Y) == N
        @test typeof(Y) == Array{Distribution, 1}

        # Test that cov and mean are finite
        bv = Array{Bool,2}(undef, N, 2)
        for k = 1:N
            bv[k, 1] = all(k -> isfinite(k), mean(Y[k]))
            bv[k, 2] = all(k -> isfinite(k), cov(Y[k]))
        end
        @test all(bv[:, 1])
        @test all(bv[:, 2])

        # Calculate the mixture model mean and covariance
        μh = sum(mean.(Y)) / N
        Σh = zeros(sys.ny, sys.ny)
        for k = 1:N
            μ_diff = mean(Y[k]) - μh
            Σh .+= μ_diff * μ_diff' + cov(Y[k])
        end
        Σh ./= N

        return μh, Σh
    end


    # Test the prediction of y via the particle filter, on an explicit LG system
    # and a random stable system with nonlinear measurements

    # True values
    μ_1 = C1*A1*mean(p0_1)
    Σ_1 = C1*A1*Q1*A1'*C1' + C1*Q1*C1' + R1

    μ_2 = C2*A2*mean(p0_2)
    Σ_2 = C2*A2*Q2*A2'*C2' + C2*Q2*C2' + R2

    μh_1, Σh_1 = test_pred(sys1, p0_1)
    μh_2, Σh_2 = test_pred(sys2, p0_2)

    # The estimated mean and covariance matrices should be fairly close to the
    # true values
    @test norm(μ_1 - μh_1) < 0.15
    @test norm(Σ_1 - Σh_1) < 0.25

    @test norm(μ_2 - μh_2) < 0.2
    @test norm(Σ_2 - Σh_2) < 0.35

    # Check that the simulated system is of the correct size and type, and that
    # no values are NaN or infinite
    x, y = sim_sys(sys2)
    @test typeof(x) == typeof(y) == Array{Float64,2}
    @test size(x) == (sys2.nx, T) && size(y) == (sys2.ny, T)
    @test all(k -> isfinite(k), x) && all(k -> isfinite(k), y)

    x, y = sim_sys(sys2, T=Float16)
    @test typeof(x) == typeof(y) == Array{Float16,2}
    @test size(x) == (sys2.nx, T) && size(y) == (sys2.ny, T)
    @test all(k -> isfinite(k), x) && all(k -> isfinite(k), y)

    x0 = 5*ones(sys2.nx)
    x, y = sim_sys(sys2, T=Float32, x0=x0)
    @test typeof(x) == typeof(y) == Array{Float32,2}
    @test size(x) == (sys2.nx, T) && size(y) == (sys2.ny, T)
    @test all(k -> isfinite(k), x) && all(k -> isfinite(k), y)
    @test x[:, 1] == x0

    # Check that our generated PFD is the same as the true one
    pfd = EP.particle_data( X = ones(Float32, 123, sys3.nx, sys3.t_end),
                            Xr = ones(Float32, 123, sys3.nx, 1),
                            Xp = ones(Float32, 123, sys3.nx, 1),
                            W = ones(Float32, 123, sys3.t_end),
                            V = ones(Float32, 123, 1),
                            H = ones(Float32, sys3.ny, 2, sys3.t_end),
                            Hh = Array{Distribution, 2}(undef, 7, 1),
                            qv_list = ones(Float32, 7, 123, 1),
                            q_list = Array{Distribution, 2}(undef, 123, 1),
                            S = ones(Int64, 123, sys3.t_end),
                            Γ = ones(Bool, sys3.t_end),
                            fail = ones(Bool, sys3.t_end),
                            p_trig = zeros(Float32, sys3.t_end))

    pfd_long = EP.particle_data( X = ones(Float32, 123, sys3.nx, sys3.t_end),
                            Xr = ones(Float32, 123, sys3.nx, sys3.t_end),
                            Xp = ones(Float32, 123, sys3.nx, sys3.t_end),
                            W = ones(Float32, 123, sys3.t_end),
                            V = ones(Float32, 123, sys3.t_end),
                            H = ones(Float32, sys3.ny, 2, sys3.t_end),
                            Hh = Array{Distribution, 2}(undef, 7, sys3.t_end),
                            qv_list = ones(Float32, 7, 123, sys3.t_end),
                            q_list = Array{Distribution, 2}(undef, 123, sys3.t_end),
                            S = ones(Int64, 123, sys3.t_end),
                            Γ = ones(Bool, sys3.t_end),
                            fail = ones(Bool, sys3.t_end),
                            p_trig = zeros(Float32, sys3.t_end))

    opt = ebpf_options(sys=sys3, N=123, kernel=kernel_IBT(zeros(sys3.ny)),
            pftype=pftype_auxiliary(x -> 0, y -> 0, 7))
    opt_long = ebpf_options(sys=sys3, N=123, kernel=kernel_IBT(zeros(sys3.ny)),
            pftype=pftype_auxiliary(x -> 0, y -> 0, 7), debug_save=true)

    pfd2 = EP.generate_pfd(opt, T=Float32)
    pfd2_long = EP.generate_pfd(opt_long, T=Float32)

    fields = fieldnames(EP.particle_data)
    pass_sizes = zeros(Bool, length(fields))
    pass_types = zeros(Bool, length(fields))
    for k = 1:length(fields)
        if fields[k] in [:extra]
            pass_sizes[k] = true
            pass_types[k] = true
        else
            pass_sizes[k] = size(getfield(pfd, fields[k])) == size(getfield(pfd2, fields[k]))
            pass_types[k] = typeof(getfield(pfd, fields[k])) == typeof(getfield(pfd2, fields[k]))
        end
    end
    @test all(pass_sizes)
    @test all(pass_types)

    fields = fieldnames(EP.particle_data)
    pass_sizes = zeros(Bool, length(fields))
    pass_types = zeros(Bool, length(fields))
    for k = 1:length(fields)
        if fields[k] in [:extra]
            pass_sizes[k] = true
            pass_types[k] = true
        else
            pass_sizes[k] = size(getfield(pfd_long, fields[k])) == size(getfield(pfd2_long, fields[k]))
            pass_types[k] = typeof(getfield(pfd_long, fields[k])) == typeof(getfield(pfd2_long, fields[k]))
        end
    end
    @test all(pass_sizes)
    @test all(pass_types)

    # Check that the symmetrization yields the correct type and size
    Σ = EP.fix_sym(Random.rand(Float16, 100, 100), warning=false)
    @test typeof(Σ) == Array{Float16,2}
    @test size(Σ) == (100,100)
    # No values should be NaN or infinite
    @test all(k -> isfinite(k), Σ)
    # It should also be symmetric
    @test issymmetric(Σ)

    # Check the error metric computations
    pfd = EP.particle_data( X = randn(Float32, 100, 5, 500*T),
                            Xr = ones(Float32, 123, sys3.nx, sys3.t_end),
                            Xp = ones(Float32, 123, sys3.nx, sys3.t_end),
                            W = ones(Float32, 100, 500*T),
                            V = ones(Float32, 123, sys3.t_end),
                            H = ones(Float32, sys3.ny, 2, sys3.t_end),
                            Hh = Array{Distribution, 2}(undef, 7, sys3.t_end),
                            qv_list = ones(Float32, 7, 123, sys3.t_end),
                            q_list = Array{Distribution, 2}(undef, 123, sys3.t_end),
                            S = ones(Int64, 123, sys3.t_end),
                            Γ = rand(Bool, 500*T),
                            fail = ones(Bool, sys3.t_end),
                            p_trig = zeros(Float32, sys3.t_end))


    x = ones(Float32, 5, 500*T)

    pfd.W[3, 123] = NaN
    pfd.W .*= 1/100
    pfd.X[:, :, findall(x -> x == 1, pfd.Γ)] .= 1

    #@test try compute_err_metrics(pfd, x); true catch; false end
    res = compute_err_metrics(pfd, x)
    @test typeof(res.err_all) == Array{Float32, 1}
    @test   size(res.err_all) == size(res.err_trig) == size(res.err_noTrig) ==
            size(res.effectiveness) == (5,)
    fields = fieldnames(EP.err_result)
    pass_finite = zeros(Bool, length(fields))
    for k = 1:length(fields)
        pass_finite[k] = all(x -> isfinite(x), getfield(res, fields[k]))
    end
    @test all(pass_finite)
    @test maximum(res.err_trig) < 1e-9
    @test maximum(abs.(res.err_noTrig .- 1.0)) < 0.05
    @test res.filtered_nans == 1


    pfd = EP.particle_data( X = randn(Float64, 100, sys5.nx, sys5.t_end),
                            Xr = ones(Float64, 123, sys5.nx, sys5.t_end),
                            Xp = ones(Float64, 123, sys5.nx, sys5.t_end),
                            W = ones(Float64, 100, sys5.t_end),
                            V = ones(Float64, 123, sys5.t_end),
                            H = ones(Float64, sys3.ny, 2, sys3.t_end),
                            Hh = Array{Distribution, 2}(undef, 7, sys3.t_end),
                            qv_list = ones(Float64, 7, 123, sys3.t_end),
                            q_list = Array{Distribution, 2}(undef, 123, sys3.t_end),
                            S = ones(Int64, 123, sys3.t_end),
                            Γ = rand(Bool, sys5.t_end),
                            fail = ones(Bool, sys3.t_end),
                            p_trig = zeros(Float64, sys3.t_end))

    x = ones(Float64, 1, sys5.t_end)
    x[[10, 50, 80]] .= 1e300

    pfd.X[:, :, findall(x -> x == 1, pfd.Γ)] .= 1
    pfd.W .*= 1/100
    res = compute_err_metrics(pfd, x)

    @test res.ce_errs == 3
    @test res.ce_trig < res.ce_noTrig

end

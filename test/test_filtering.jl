@testset "test_filtering" begin

    function output_tests(output, opt, x)

        # None of the outputs should have NaN of Inf values
        fields = fieldnames(EP.particle_data)
        pass = zeros(Bool, length(fields))
        for k = 1:length(fields)
            if fields[k] in [:Hh, :q_list, :extra]
                pass[k] = true
            else
                pass[k] = all(x -> isfinite(x), getfield(output, fields[k]))
            end
        end
        @test all(pass)

        # The weights should be normalized and positive
        @test all(output.W .>= 0)
        @test all(x -> isapprox(x, 1.0), sum(output.W, dims=1))

        idx_0 = findall(x -> x == 0, output.Γ) # Find the non-triggered time instances
        idx_1 = findall(x -> x == 1, output.Γ) # Find all triggered events

        xh = zeros(opt.sys.nx, opt.sys.t_end)
        Ph = zeros(opt.sys.nx, opt.sys.nx, opt.sys.t_end)
        for k = 1:opt.sys.t_end
            xh[:, k] = mean(output.X[:, :, k], Weights(output.W[:, k]), dims=1)
            Ph[:,:,k] = cov(output.X[:, :, k], Weights(output.W[:, k]), corrected=false)
        end

        return xh, Ph

    end

    function run_tests_linear(sys_type, N, T)
        sys, A, C, Q, R, p0 = EP.generate_example_system(T, type=sys_type)
        x, y = sim_sys(sys, x0=rand(p0))

        q, qv = EP.genprop_linear_gaussian_noise()

        xh_kalman, Ph_kalman = EP.kalman_filter(y, A, C, Q, R, sys)

        if sys.ny > 1
            LH = likelihood_MC(M=1)
        else
            LH = likelihood_analytic()
        end

        # Test baseline bootstrap filter
        println("Testing baseline bootstrap filter")
        opt = ebpf_options(sys=sys, N=10000, kernel=kernel_SOD(zeros(sys.ny)),
            pftype=pftype_bootstrap(), triggerat="always")
        output = ebpf(y, opt, X0=rand(p0, opt.N)')

        xh_bpf_good, Ph_bpf_good = output_tests(output, opt, x)

        # Test bootstrap filter
        println("Testing bootstrap filter")
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(zeros(sys.ny)),
            pftype=pftype_bootstrap(),
            likelihood=LH)
        output = ebpf(y, opt, X0=rand(p0, opt.N)')

        xh_bpf, Ph_bpf = output_tests(output, opt, x)

        # Test bootstrap filter with SOD
        println("Testing bootstrap filter with SOD")

        δ = (maximum(y, dims=2) - minimum(y, dims=2))[:] ./ 4
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(δ),
            pftype=pftype_bootstrap(),
            likelihood=LH)
        output = ebpf(y, opt, X0=rand(p0, opt.N)')

        xh_bpf_sod, Ph_bpf_sod = output_tests(output, opt, x)
        p_trig_bpf_sod = copy(output.p_trig)

        #Test bootstrap filter with IBT
        println("Testing bootstrap filter with IBT")

        δ = (maximum(y, dims=2) - minimum(y, dims=2))[:] ./ 4
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT(δ),
            pftype=pftype_bootstrap(),
            likelihood=LH)
        output = ebpf(y, opt, X0=rand(p0, opt.N)')

        xh_bpf_ibt, Ph_bpf_ibt = output_tests(output, opt, x)
        p_trig_bpf_ibt = copy(output.p_trig)

        if sys.ny == 1
            # Test auxiliary filter
            println("Testing auxiliary filter")
            opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD([0.0]),
                pftype=pftype_auxiliary(qv, q, 5), triggerat="always")
            output = ebpf(y, opt, X0=rand(p0, opt.N)')

            xh_apf, Ph_apf = output_tests(output, opt, x)

            # Test auxiliary filter with SOD
            println("Testing auxiliary filter with SOD")

            δ = (maximum(y, dims=2) - minimum(y, dims=2))[:] ./ 4
            opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD(δ),
                pftype=pftype_auxiliary(qv, q, 5))
            output = ebpf(y, opt, X0=rand(p0, opt.N)')

            xh_apf_sod, Ph_apf_sod = output_tests(output, opt, x)
            p_trig_apf_sod = copy(output.p_trig)

            #Test auxiliary filter with IBT
            println("Testing auxiliary filter with IBT")

            δ = (maximum(y, dims=2) - minimum(y, dims=2))[:] ./ 4
            opt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT(δ),
                pftype=pftype_auxiliary(qv, q, 5))
            output = ebpf(y, opt, X0=rand(p0, opt.N)')

            xh_apf_ibt, Ph_apf_ibt = output_tests(output, opt, x)
            p_trig_apf_ibt = copy(output.p_trig)
        end

        # Kalman filter and PF should yield roughly the same estimates of x and P
        @test maximum(mean((xh_kalman - xh_bpf_good).^2, dims=2)) < 0.3
        @test maximum(mean((Ph_kalman - Ph_bpf_good).^2, dims=3)) < 0.8

        # The error from SOD should be larger than no trigger
        @test maximum(mean((xh_kalman - xh_bpf_sod).^2, dims=2)) >
                maximum(mean((xh_kalman - xh_bpf).^2, dims=2))
        @test maximum(mean((Ph_kalman - Ph_bpf_sod).^2, dims=3)) >
                maximum(mean((Ph_kalman - Ph_bpf).^2, dims=3))

        # The error from IBT should be larger than no trigger
        @test maximum(mean((xh_kalman - xh_bpf_ibt).^2, dims=2)) >
                maximum(mean((xh_kalman - xh_bpf).^2, dims=2))
        @test maximum(mean((Ph_kalman - Ph_bpf_ibt).^2, dims=3)) >
                maximum(mean((Ph_kalman - Ph_bpf).^2, dims=3))

        # The event-sampled covariance should be larger than no trigger
        tr_bpf_sod = [tr(Ph_bpf_sod[:,:,k]) for k in 1:T]
        tr_bpf_ibt = [tr(Ph_bpf_ibt[:,:,k]) for k in 1:T]
        tr_bpf = [tr(Ph_bpf[:,:,k]) for k in 1:T]
        @test mean(tr_bpf_sod) > mean(tr_bpf)
        @test mean(tr_bpf_ibt) > mean(tr_bpf)

        if sys.ny == 1
            @test maximum(mean((xh_kalman - xh_apf).^2, dims=2)) < 0.1
            @test maximum(mean((Ph_kalman - Ph_apf).^2, dims=3)) < 0.5

            @test maximum(mean((xh_kalman - xh_apf_sod).^2, dims=2)) >
                    maximum(mean((xh_kalman - xh_apf).^2, dims=2))
            @test maximum(mean((Ph_kalman - Ph_apf_sod).^2, dims=3)) >
                    maximum(mean((Ph_kalman - Ph_apf).^2, dims=3))

            @test maximum(mean((xh_kalman - xh_apf_ibt).^2, dims=2)) >
                    maximum(mean((xh_kalman - xh_apf).^2, dims=2))
            @test maximum(mean((Ph_kalman - Ph_apf_ibt).^2, dims=3)) >
                    maximum(mean((Ph_kalman - Ph_apf).^2, dims=3))

            tr_apf_sod = [tr(Ph_apf_sod[:,:,k]) for k in 1:T]
            tr_apf_ibt = [tr(Ph_apf_ibt[:,:,k]) for k in 1:T]
            tr_apf = [tr(Ph_apf[:,:,k]) for k in 1:T]
            @test mean(tr_apf_sod) > mean(tr_apf)
            @test mean(tr_apf_ibt) > mean(tr_apf)

            # APF should be better than BPF
            @test maximum(mean((xh_kalman - xh_apf).^2, dims=2)) <
                    maximum(mean((xh_kalman - xh_bpf).^2, dims=2))
            @test maximum(mean((Ph_kalman - Ph_apf).^2, dims=3)) <
                    maximum(mean((Ph_kalman - Ph_bpf).^2, dims=3))

        end

        # Probability triggers should be nonzero, and smaller than 1
        @test all(0 .<= p_trig_bpf_sod[2:end] .<= 1)
        @test all(0 .<= p_trig_bpf_ibt[2:end] .<= 1)
        if sys.ny == 1
            @test all(0 .<= p_trig_apf_sod[2:end] .<= 1)
            @test all(0 .<= p_trig_apf_ibt[2:end] .<= 1)
        end

    end

    function run_tests_nonlinear(sys_type, N, T)

        sys, A, C, Q, R, p0 = EP.generate_example_system(T, type=sys_type)
        x, y = sim_sys(sys, x0=rand(p0))

        q, qv = EP.genprop_linear_gaussian_noise()

        # Test baseline bootstrap filter
        println("Testing bootstrap")
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD([0.0]),
            pftype=pftype_bootstrap(), triggerat="always")
        output = ebpf(y, opt, X0=rand(p0, N)')
        xh_bpf, Ph_bpf = output_tests(output, opt, x)
        res_bpf = compute_err_metrics(output, x)

        # Test bootstrap filter with SOD
        println("Testing bootstrap with SOD")
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD([10.0]),
            pftype=pftype_bootstrap())
        output = ebpf(y, opt, X0=rand(p0, N)')
        xh_bpf_sod, Ph_bpf_sod = output_tests(output, opt, x)
        res_bpf_sod = compute_err_metrics(output, x)
        p_trig_bpf_sod = copy(output.p_trig)


        #Test bootstrap filter with IBT
        println("Testing bootstrap with IBT")
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([3.0]),
            pftype=pftype_bootstrap())
        output = ebpf(y, opt, X0=rand(p0, N)')
        xh_bpf_ibt, Ph_bpf_ibt = output_tests(output, opt, x)
        res_bpf_ibt = compute_err_metrics(output, x)
        p_trig_bpf_ibt = copy(output.p_trig)

        # Test baseline auxiliary filter
        println("Testing auxiliary")
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD([0.0]),
            pftype=pftype_auxiliary(qv, q, 5))
        output = ebpf(y, opt, X0=rand(p0, N)')
        xh_apf, Ph_apf = output_tests(output, opt, x)
        res_apf = compute_err_metrics(output, x)


        # Test auxiliary filter with SOD
        println("Testing auxiliary with SOD")
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_SOD([10.0]),
            pftype=pftype_auxiliary(qv, q, 5))
        output = ebpf(y, opt, X0=rand(p0, N)')
        xh_apf_sod, Ph_apf_sod = output_tests(output, opt, x)
        res_apf_sod = compute_err_metrics(output, x)
        p_trig_apf_sod = copy(output.p_trig)

        #Test auxiliary filter with IBT
        println("Testing auxiliary with IBT")
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([3.0]),
            pftype=pftype_auxiliary(qv, q, 5))
        output = ebpf(y, opt, X0=rand(p0, N)')
        xh_apf_ibt, Ph_apf_ibt = output_tests(output, opt, x)
        res_apf_ibt = compute_err_metrics(output, x)
        p_trig_apf_ibt = copy(output.p_trig)
        # Test the results

        # IBT should be better than SOD with these settings
        @test maximum(res_bpf_sod.err_all) >
            maximum(res_bpf_ibt.err_all)
        @test maximum(res_apf_sod.err_all) >
            maximum(res_apf_ibt.err_all)

        # The cross-entropy should be better at both trigger instances and for IBT.
        # BPF should produce more errors than APF
        @test res_bpf_sod.ce_trig > res_apf_sod.ce_trig
        @test res_bpf_ibt.ce_trig > res_apf_ibt.ce_trig
        @test res_bpf_sod.ce_all > res_apf_sod.ce_all
        @test res_bpf_ibt.ce_all > res_apf_ibt.ce_all

        # The event-sampled covariance should be larger than the baseline
        tr_bpf = [tr(Ph_bpf[:,:,k]) for k in 1:T]
        tr_bpf_sod = [tr(Ph_bpf_sod[:,:,k]) for k in 1:T]
        tr_bpf_ibt = [tr(Ph_bpf_ibt[:,:,k]) for k in 1:T]
        tr_apf = [tr(Ph_apf[:,:,k]) for k in 1:T]
        tr_apf_sod = [tr(Ph_apf_sod[:,:,k]) for k in 1:T]
        tr_apf_ibt = [tr(Ph_apf_ibt[:,:,k]) for k in 1:T]

        @test mean(tr_bpf_sod) > mean(tr_bpf)
        @test mean(tr_bpf_ibt) > mean(tr_bpf)

        @test mean(tr_apf_sod) > mean(tr_apf)
        @test mean(tr_apf_ibt) > mean(tr_apf)

        # Probability triggers should be nonzero, and smaller than 1
        @test all(0 .<= p_trig_bpf_sod .<= 1)
        @test all(0 .<= p_trig_bpf_ibt .<= 1)
        @test all(0 .<= p_trig_apf_sod .<= 1)
        @test all(0 .<= p_trig_apf_ibt .<= 1)

        # Test that debug_save gives the same result

        println("Testing debug_save")
        Random.seed!(123)
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([3.0]),
            pftype=pftype_auxiliary(qv, q, 5))
        output = ebpf(y, opt, X0=rand(p0, N)')

        Random.seed!(123)
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([3.0]),
            pftype=pftype_auxiliary(qv, q, 5), debug_save=true)
        output2 = ebpf(y, opt, X0=rand(p0, N)')

        @test isapprox(sum(output.X - output2.X), 0)

        # Test that the predicive computations yields roughly the same as
        # the standard ebpf

        println("Testing EBPF with predictive computation")
        opt = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([3.0]),
            pftype=pftype_auxiliary(qv, q, 5),
            predictive_computation=true,
            extra_params=Dict("a" => 0.80))
        output = ebpf(y, opt, X0=rand(p0, N)')

        xh_predcomp, Ph_predcomp = output_tests(output, opt, x)
        res_predcomp = compute_err_metrics(output, x)

        t = output.extra["triggerwhen"]
        @test 0.8 < 1 - sum(t[:, 1] .== t[:, 2]) / size(t, 1) < 0.95
        @test abs(res_predcomp.ce_all - res_apf_ibt.ce_all) < 0.1

    end

    Random.seed!(123)

    T = 1000

    printstyled("Testing on linear 1D output system\n", bold=true, color=:green)
    run_tests_linear("linear_ny1", 50, T)

    printstyled("Testing on linear 2D output system\n", bold=true, color=:green)
    run_tests_linear("linear_ny2", 50, T)

    printstyled("Testing on nonlinear 1D classic system\n", bold=true, color=:green)
    run_tests_nonlinear("nonlinear_classic", 100, T)

end

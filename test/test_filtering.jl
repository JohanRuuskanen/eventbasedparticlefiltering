@testset "test_filtering" begin

    function local_tests(output, par)

        # Check that the sizes of the outputs are correct
        @test size(output.X) == (N, nx, T)
        @test size(output.W) == (N, T)
        @test size(output.S) == (N, T)
        @test size(output.Z) == (ny, T)
        @test size(output.Γ) == (T,)
        @test size(output.res) == (T,)
        @test size(output.fail) == (T,)

        # None of the outputs should have NaN of Inf values
        @test all(x -> isfinite(x), output.X)
        @test all(x -> isfinite(x), output.W)
        @test all(x -> isfinite(x), output.S)
        @test all(x -> isfinite(x), output.Z)
        @test all(x -> isfinite(x), output.Γ)
        @test all(x -> isfinite(x), output.res)
        @test all(x -> isfinite(x), output.fail)

        # The logic arrays only contains 0 or 1.
        @test all(x -> x in union(0,1), output.Γ)
        @test all(x -> x in union(0,1), output.res)
        @test all(x -> x in union(0,1), output.fail)

        # The amount of resamples should neither be no resamples, or resample at
        # each time step
        @test 0 < sum(output.res) < T

        # The weights should be normalized
        @test all(x -> isapprox(x, 1.0), sum(output.W, dims=1))

        idx_0 = findall(x -> x == 0, output.Γ) # Find the non-triggered time instances
        idx_1 = findall(x -> x == 1, output.Γ) # Find all triggered events

        xh = zeros(nx, T)
        for k = 1:T
            xh[:, k] = output.W[:, k]' * output.X[:,:,k]
        end
        r = x - xh

        mse_0 = mean(r[:, idx_0].^2, dims=2)
        mse_1 = mean(r[:, idx_1].^2, dims=2)

        if sum(output.Γ) < T-1
            # Test that time steps with triggered events have a smaller MSE than
            # time steps that have no triggered events.
            @test all(x -> x, [mse_0[i] > mse_1[i] for i = 1:nx])

            # Check that time steps with no triggered events have a smaller MSE
            # than a dummy value
            @test all(x -> x < 6, mse_0)
        end

        # Check that time steps with triggered events have a smaller MSE
        # than a dummy value
        @test all(x -> x < 2, mse_1)

        return idx_1
    end

    Random.seed!(123)

    # Parameters
    N = 200
    T = 1000
    δ = 4

    A = [0.8 1; 0 0.95]
    C = [0.7 0.6]

    nx = size(A, 1)
    ny = size(C, 1)

    Q = 1*Matrix{Float64}(I, nx, nx)
    R = 0.1*Matrix{Float64}(I, ny, ny)

    sys = sys_params(A, C, Q, R, T)
    x, y = sim_sys(sys)

    # Test APF, BPF with periodic sampling
    println("Testing ordinary BPF")
    par = pf_params(N, "SOD", N/2)
    output_bpf = ebpf(y, sys, par, 0)
    local_tests(output_bpf, par)

    println("Testing ordinary APF")
    par = pf_params(N, "SOD", N/2)
    output_apf = eapf(y, sys, par, 0)
    local_tests(output_apf, par)

    # In this case, events are generated at every time instance (except for the
    # first)
    @test sum(output_bpf.Γ) == T - 1
    @test sum(output_apf.Γ) == T - 1

    # Test BPF with SOD sampling
    println("Testing BPF SOD")
    par = pf_params(N, "SOD", N/2)
    output = ebpf(y, sys, par, δ)
    idx = local_tests(output, par)

    # The effective sample size for BPF at new events should be lower than some
    # low dummy value
    Neff = 1 ./ sum(output.W[:, idx].^2, dims=1)
    @test mean(Neff) < (par.N/4)

    # Test APF wuth SOD sampling
    println("Testing APF SOD")
    par = pf_params(N, "SOD", N/2)
    output = eapf(y, sys, par, δ)
    idx = local_tests(output, par)

    # The effective sample size for APF at new events should be higher than some
    # high dummy value
    Neff = 1 ./ sum(output.W[:, idx].^2, dims=1)
    @test mean(Neff) > (par.N/4 * 3)

    # Test BPF with MBT sampling
    println("Testing BPF MBT")
    par = pf_params(N, "MBT", N/2)
    output = ebpf(y, sys, par, δ)
    idx = local_tests(output, par)

    # The effective sample size for BPF at new events should be lower than some
    # low dummy value
    Neff = 1 ./ sum(output.W[:, idx].^2, dims=1)
    @test mean(Neff) < (par.N/4)

    # Test APF with MBT sampling
    println("Testing APF MBT")
    par = pf_params(N, "MBT", N/2)
    output = eapf(y, sys, par, δ)
    idx = local_tests(output, par)

    # The effective sample size for APF at new events should be higher than some
    # high dummy value
    Neff = 1 ./ sum(output.W[:, idx].^2, dims=1)
    @test mean(Neff) > (par.N/4 * 3)

end

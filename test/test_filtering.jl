@testset "test_filtering" begin

    function local_tests(output, par)
        @test size(output.X) == (N, nx, T)
        @test size(output.W) == (N, T)
        @test size(output.S) == (N, T)
        @test size(output.Z) == (ny, T)
        @test size(output.Γ) == (T,)
        @test size(output.res) == (T,)
        @test size(output.fail) == (T,)

        @test all(x -> isfinite(x), output.X)
        @test all(x -> isfinite(x), output.W)
        @test all(x -> isfinite(x), output.S)
        @test all(x -> isfinite(x), output.Z)
        @test all(x -> isfinite(x), output.Γ)
        @test all(x -> isfinite(x), output.res)
        @test all(x -> isfinite(x), output.fail)

        @test all(x -> x in union(0,1), output.Γ)
        @test all(x -> x in union(0,1), output.res)
        @test all(x -> x in union(0,1), output.fail)

        @test 0 < sum(output.res) < T

        @test all(x -> isapprox(x, 1.0), sum(output.W, dims=1))

        idx_0 = findall(x -> x == 0, output.Γ)
        idx_1 = findall(x -> x == 1, output.Γ)

        xh = zeros(nx, T)
        for k = 1:T
            xh[:, k] = output.W[:, k]' * output.X[:,:,k]
        end
        r = x - xh

        mse_0 = mean(r[:, idx_0].^2, dims=2)
        mse_1 = mean(r[:, idx_1].^2, dims=2)

        if sum(output.Γ) < T-1
            @test all(x -> x, [mse_0[i] > mse_1[i] for i = 1:nx])
            @test all(x -> x < 6, mse_0)
        end

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

    # Test oridnary APF, BPF
    println("Testing ordinary BPF")
    par = pf_params(N, "SOD", N/2)
    output = ebpf(y, sys, par, 0)
    idx = local_tests(output, par)
    @test sum(output.Γ) == T - 1

    println("Testing ordinary APF")
    par = pf_params(N, "SOD", N/2)
    output = eapf(y, sys, par, 0)
    idx = local_tests(output, par)
    @test sum(output.Γ) == T - 1

    # Test BPF SOD
    println("Testing BPF SOD")
    par = pf_params(N, "SOD", N/2)
    output = ebpf(y, sys, par, δ)
    idx = local_tests(output, par)

    Neff = 1 ./ sum(output.W[:, idx].^2, dims=1)
    @test mean(Neff) < (par.N/4)

    # Test APF SOD
    println("Testing APF SOD")
    par = pf_params(N, "SOD", N/2)
    output = eapf(y, sys, par, δ)
    idx = local_tests(output, par)

    Neff = 1 ./ sum(output.W[:, idx].^2, dims=1)
    @test mean(Neff) > (par.N/4 * 3)

    # Test BPF MBT
    println("Testing BPF MBT")
    par = pf_params(N, "MBT", N/2)
    output = ebpf(y, sys, par, δ)
    idx = local_tests(output, par)

    Neff = 1 ./ sum(output.W[:, idx].^2, dims=1)
    @test mean(Neff) < (par.N/4)

    # Test APF MBT
    println("Testing APF MBT")
    par = pf_params(N, "MBT", N/2)
    output = eapf(y, sys, par, δ)
    idx = local_tests(output, par)

    Neff = 1 ./ sum(output.W[:, idx].^2, dims=1)
    @test mean(Neff) > (par.N/4 * 3)

end

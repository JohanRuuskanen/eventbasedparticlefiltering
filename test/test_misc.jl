@testset "test_misc" begin

    Random.seed!(123)

    function create_params(m, n, T)
        A = Random.rand(m, m); C = Random.rand(n, m)
        Q = 1*Matrix{Float64}(I, m, m)
        R = 0.1*Matrix{Float64}(I, n, n)

        sys = try sys_params(A, C, Q, R, T) catch; false end
        return sys
    end

    # Parameters
    m = 5; n = 4; T = 200;
    sys = create_params(m, n, T)

    # It should be possible to create a system struct
    @test sys != false

    # It should be possible to create a particle filter parameters struct
    @test try pf_params(123, "this is a string", 12.34); true catch; false end

    # It should be possible to create an output struct
    X = Random.rand!
    @test try output(ones(3,3,3), ones(3,3), ones(2,2), ones(2,2), ones(2),
        ones(2), ones(2)); true catch; false end

    # It should be possible to simulate a system
    @test try sim_sys(sys); true catch; false end

    # It should be possible to fix the symmetry of a matrix
    @test try EP.fix_sym(ones(n, n)); true catch; false end

    x, y = sim_sys(sys)
    # Check that the simulated system is of the correct size and type
    @test typeof(x) == typeof(y) == Array{Float64,2}
    @test size(x) == (m, T)
    @test size(y) == (n, T)
    # No values should be NaN or infinite
    @test all(k -> isfinite(k), x)
    @test all(k -> isfinite(k), y)

    sys = create_params(1, 1, T)
    x, y = sim_sys(sys)
    # A one dimensional system should still yield matrix outputs
    @test typeof(x) == typeof(y) == Array{Float64,2}

    Σ = EP.fix_sym(Random.rand(m, m), warning=false)
    # Check that the symmetrization yields the correct type and size
    @test typeof(Σ) == Array{Float64,2}
    @test size(Σ) == (m,m)
    # No values should be NaN or infinite
    @test all(k -> isfinite(k), Σ)
    # It should also be symmetric
    @test issymmetric(Σ)


end

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

    @test sys != false

    @test try pf_params(123, "this is a string", 12.34); true catch; false end

    X = Random.rand!
    @test try output(ones(3,3,3), ones(3,3), ones(2,2), ones(2,2), ones(2),
        ones(2), ones(2)); true catch; false end

    @test try sim_sys(sys); true catch; false end

    @test try fix_sym(ones(n, n)); true catch; false end

    x, y = sim_sys(sys)
    @test typeof(x) == typeof(y) == Array{Float64,2}
    @test size(x) == (m, T)
    @test size(y) == (n, T)

    sys = create_params(1, 1, T)
    x, y = sim_sys(sys)
    @test typeof(x) == typeof(y) == Array{Float64,2}
    @test size(x) == (1, T)
    @test size(y) == (1, T)
    @test isempty(findall(k -> isnan(k), x))
    @test isempty(findall(k -> isnan(k), y))

    Σ = fix_sym(Random.rand(m, m), warning=false)
    @test typeof(Σ) == Array{Float64,2}
    @test size(Σ) == (m,m)
    @test issymmetric(Σ)
    @test isempty(findall(k -> isnan(k), Σ))

end

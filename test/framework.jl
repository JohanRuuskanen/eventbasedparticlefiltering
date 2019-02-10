
function run_tests(my_tests)
    @testset "All tests" begin
        for test in my_tests
            println(test)
            include("$(test).jl")
        end
    end
end

function test_script_crash(script_path)
    try
        include("$(script_path)")
        return true
    catch
        return false
    end
end

function create_params(m, n, T)
    # Create a stable system
    A = 0.2 * randn(m, m)
    while !isempty(findall(x-> x >= 0.95, abs.(eigvals(A))))
        A = 0.2 * randn(m, m)
    end

    C = Random.rand(n, m)
    Q = 1*Matrix{Float64}(I, m, m)
    R = 0.1*Matrix{Float64}(I, n, n)

    sys = try sys_params(A, C, Q, R, T) catch; false end
    return sys
end


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
    catch e
        println(e)
        return false
    end
end

function estimate_output_monte_carlo(sys, p0; MC = 10000)
    y = zeros(sys.ny, MC)
    for k = 1:MC
        x0 = rand(p0)
        _, ytmp = sim_sys(sys, T=Float32, x0=x0)
        y[:, k] = ytmp[:, 2]
    end
    return mean(y, dims=2)[:], cov(y, dims=2)
end

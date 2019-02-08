
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
        plt[:close]("all")
        return true
    catch
        plt[:close]("all")
        return false
    end
end
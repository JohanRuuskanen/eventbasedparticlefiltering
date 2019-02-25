@testset "test_examples" begin

current_path = @__DIR__
example_path = current_path[1:end-4] * "examples/"
scripts = readdir(example_path)

for script in scripts
    println("Testing "*script)

    # For each example, test that it doesn't crash
    @test test_script_crash(example_path * script)
end

end

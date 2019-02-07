@testset "test_examples" begin

example_path = "../examples/"
scripts = readdir(example_path)

for script in scripts
    @test test_script_crash(example_path * script)
end

end

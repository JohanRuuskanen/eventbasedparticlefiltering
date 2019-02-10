@testset "test_plotting" begin

Random.seed!(123)

m = 10; n = 50;
X = Random.rand(m, n); W = Random.rand(m, n); S = Random.rand(m, n);
x_true = Random.rand(1, n); y = Random.rand(n); z = Random.rand(n)
Γ = Random.rand([0, 1], n); δ = 10

@test try plot_particle_trace(X, S, x_true=x_true, Γ=Γ); true catch; false end

@test try plot_data(y, z, δ=δ); true catch; false end

@test try plot_effective_sample_size(W; Γ=Γ); true catch; false end

end

@testset "test_predcomp" begin

    q, qv = EP.genprop_linear_gaussian_noise()

    Random.seed!(123)

    MC = 1000
    N = 1000
    T = 100

    sys, A, C, Q, R, p0 = EP.generate_example_system(T, type="nonlinear_classic")
    x, y = sim_sys(sys, x0=rand(p0))

    δ = [5] #(maximum(y, dims=2) - minimum(y, dims=2))[:] ./ 2
    opt1 = ebpf_options(sys=sys, N=N, kernel=kernel_IBT(δ),
        pftype=pftype_bootstrap(), triggerat="never")
    opt2 = ebpf_options(sys=sys, N=N, kernel=kernel_IBT(δ),
        pftype=pftype_auxiliary(qv, q, 5), triggerat="never")

    # Calculate trig prob
    println("")
    output_bpf = ebpf(y, opt1, X0=rand(p0, N)')

    @test all(0 .<= output_bpf.p_trig .<= 1)

    p_first_bpf = zeros(T)
    for k = 1:T
        p_first_bpf[k] = output_bpf.p_trig[k]
        for i = 1:(k-1)
            p_first_bpf[k] *= (1 - output_bpf.p_trig[k-i])
        end
    end

    println("")
    output_apf = ebpf(y, opt2, X0=rand(p0, N)')

    @test all(0 .<= output_apf.p_trig .<= 1)

    p_first_apf = zeros(T)
    for k = 1:T
        p_first_apf[k] = output_apf.p_trig[k]
        for i = 1:(k-1)
            p_first_apf[k] *= (1 - output_apf.p_trig[k-i])
        end
    end

    # Calculate trig prob via MC
    opt1.triggerat = "events"
    opt1.abort_at_trig=true
    opt1.print_progress=false

    MC_trig = zeros(MC)
    all_trigprob = zeros(Bool, MC)
    println("")
    for k = 1:MC
        print("Running $(k) / $(MC) \r")
        flush(stdout)
        x, y = sim_sys(sys, x0=rand(p0))

        output_sim = ebpf(y, opt1, X0=rand(p0, N)')
        Γ = output_sim.Γ
        Γ[1] = 0
        val = findfirst(x -> x==1, Γ)
        if val == nothing
            val = T
        end
        MC_trig[k] = val
        all_trigprob[k] = all(0 .<= output_sim.p_trig .<= 1)
    end
    u = countmap(MC_trig)
    val = [i for i = 1:T]
    occ = [if haskey(u, i); u[i]; else; 0; end for i = 1:T]
    e_first = occ ./ MC

    # Check that all triggering outputs are valid
    @test all(all_trigprob)

    # Test that calculated triggering is roughly the same as the MC simulation
    @test sum(abs, e_first - p_first_bpf) < 0.5
    @test sum(abs, e_first - p_first_apf) < 0.5

end

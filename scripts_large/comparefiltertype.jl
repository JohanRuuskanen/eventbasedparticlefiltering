
using JLD
using Random
using PyPlot
using Distributions
using LinearAlgebra
using EventBasedParticleFiltering

const EP = EventBasedParticleFiltering

Random.seed!(2)

savepath = "/path/to/save" # Change this

## -------------------- DECLARE COMMON VARIABLES ----------------------------

sims = 1

@everywhere begin

    using Random
    using Distributions
    using LinearAlgebra
    using EventBasedParticleFiltering


    const EP = EventBasedParticleFiltering

    # Reduce the parameter sweep if running on a single computer

    D_vec = [3]
    N_vec = [25, 50, 100, 200, 400]
    LH_evals = [likelihood_analytic(), likelihood_MC(M=1)]
    T = 100000
    Δ = vcat(   collect(range(0, stop=2.5, length=100)),
                collect(range(2.5, stop=12, length=100)))

    system = "nonlinear_classic"

    sys, _, _, _, _, p0 = EP.generate_example_system(T, type=system)

    q, qv = EP.genprop_linear_gaussian_noise()

end

@everywhere function simfunc(index, params, MCsims)
    println("Running $(index) / $(MCsims)")

    if params["filtertype"] == "bpf"
        opt = ebpf_options(sys=sys, N=params["N"], kernel=kernel_IBT([params["δ"]]),
            pftype=pftype_bootstrap(), likelihood=params["lh_eval"],
            print_progress=false)
    elseif params["filtertype"] == "apf"
        opt = ebpf_options(sys=sys, N=params["N"], kernel=kernel_IBT([params["δ"]]),
            pftype=pftype_auxiliary(qv=qv, q=q, D=params["D"]),
            likelihood=params["lh_eval"], print_progress=false)
    else
        error("No such filter")
    end

    Random.seed!(index*6516 + 6612912)

    x, y = sim_sys(sys, x0=rand(p0))
    t = @elapsed begin
        output_sim = ebpf(y, opt, X0=rand(p0, params["N"])')
    end

    return Dict("error_metric" => compute_err_metrics(output_sim, x),
                "fails" => output_sim.fail,
                "time" => t)
end

## -------------------- RUN DISTRIBUTED COMPUTING ----------------------------

printstyled("Running distributed task on all workers\n"
    ,bold=true,color=:magenta)

# Splat the parameters into a single parameter vector
par_vec = Array{Dict, 1}(undef, 0)
for N in N_vec
    global par_vec
    for δ in Δ
        for lh_eval in LH_evals
            for sim = 1:sims
                par_vec = vcat(par_vec, Dict("filtertype" => "bpf",
                                                "N" => N,
                                                "δ" => δ,
                                                "lh_eval" => lh_eval,
                                                "sim" => sim,
                                                "D" => 1))
                for D in D_vec
                    par_vec = vcat(par_vec, Dict("filtertype" => "apf",
                                                    "N" => N,
                                                    "δ" => δ,
                                                    "lh_eval" => lh_eval,
                                                    "sim" => sim,
                                                    "D" => D))
                end
            end
        end
    end
end

MCsims = size(par_vec, 1)
# Run the simulation

result = pmap(1:MCsims) do index
    result = simfunc(index, par_vec[index], MCsims)
end

printstyled("All tasks complete!\n",bold=true,color=:magenta)

save(savepath*"comparefiltertype.jld",
        "par_vec", par_vec,
        "result", result,
        "delta", Δ)

## -------------------- PLOT RESULTS ----------------------------


result = load(savepath*"comparefiltertype.jld", "result")
par_vec = load(savepath*"comparefiltertype.jld", "par_vec")

function extract_err_over_com(result, par_vec, filtertype, lhtype;
    x_field=:nbr_frac, y_field=:err_all)

    params = Array{Any, 2}(undef, length(par_vec), length(keys(par_vec[1])))
    for k = 1:length(par_vec)
        params[k, 1] = par_vec[k]["filtertype"]
        params[k, 2] = par_vec[k]["lh_eval"]
        params[k, 3] = par_vec[k]["N"]
        params[k, 4] = par_vec[k]["δ"]
        params[k, 5] = par_vec[k]["sim"]
        params[k, 6] = par_vec[k]["D"]
    end

    idx_filtertype = findall(x -> x == filtertype, params[:, 1])
    idx_lhtype = findall(x -> typeof(x) <: lhtype, params[:, 2])

    idx_type = intersect(idx_filtertype, idx_lhtype)
    p_type = params[idx_type, :]
    r_type = result[idx_type]

    N_vec = unique(p_type[:, 3])
    Δ = unique(p_type[:, 4])

    x = zeros(length(N_vec), length(Δ))
    y = zeros(length(N_vec), length(Δ))
    t = zeros(length(N_vec), length(Δ))
    f = zeros(length(N_vec), length(Δ))

    for (k, N) in enumerate(N_vec)
        idx_N = findall(x -> x == N, p_type[:, 3])
        r_type_N = r_type[idx_N]
        p_type_N = p_type[idx_N, :]
        for (i, δ) in enumerate(Δ)
            idx_δ = findall(x -> x == δ, p_type_N[:, 4])
            r_mc = r_type_N[idx_δ]

            data_x = zeros(length(r_mc))
            data_y = zeros(length(r_mc))
            data_t = zeros(length(r_mc))
            data_f = zeros(length(r_mc))
            for (k, res) in enumerate(r_mc)
                x_v = getfield(res["error_metric"], x_field)
                y_v = getfield(res["error_metric"], y_field)

                typeof(x_v) <: Number ? x_v : x_v = first(x_v)
                typeof(y_v) <: Number ? y_v : y_v = first(y_v)

                data_x[k] = x_v
                data_y[k] = y_v
                data_t[k] = res["time"] / 100000
                data_f[k] = sum(res["fails"]) / 100000
            end

            x[k, i] = mean(data_x)
            y[k, i] = mean(data_y)
            t[k, i] = mean(data_t)
            f[k, i] = mean(data_f)

        end
    end

    return x, y, t, f, Δ, N_vec
end

x_bpf, _, _, _, Δ_bpf, N_vec_bpf = extract_err_over_com(result, par_vec, "bpf", likelihood_analytic)
_, _, _, _, Δ_apf, N_vec_apf = extract_err_over_com(result, par_vec, "apf", likelihood_analytic)

if Δ_bpf == Δ_apf && N_vec_bpf == N_vec_apf
    Δ = Δ_bpf
    N_vec = N_vec_bpf
else
    error("Not equal!")
end

# Plot as ellipses instead to cover uncertainty in CR estimation?

lh = likelihood_analytic
N_plot = 3
δ_plot = 100

println("====")
println("deltaplot: $δ_plot is δ=$(Δ[δ_plot]) and CR=$(mean(x_bpf[:, δ_plot]))")

figure(1)
clf()
subplot(2, 3, 1)
X_all = zeros(length(Δ), 4)
Y_all = zeros(length(Δ), 4)
for (i, k) in enumerate([likelihood_analytic, likelihood_MC])
    x_bpf, y_bpf, _, _, _, _ = extract_err_over_com(result, par_vec, "bpf", k, y_field=:ce_all)
    x_apf, y_apf, _, _, _, _ = extract_err_over_com(result, par_vec, "apf", k, y_field=:ce_all)
    plot(x_bpf[N_plot, :], y_bpf[N_plot, :], "C$(i-1)*", label="bpf ($(join(["i" for tmp = 1:i])))")
    plot(x_apf[N_plot, :], y_apf[N_plot, :], "C$(i-1)s", label="apf ($(join(["i" for tmp = 1:i])))")
    X_all[:, i] = x_bpf[N_plot, :]
    X_all[:, i+2] = x_apf[N_plot, :]
    Y_all[:, i] = y_bpf[N_plot, :]
    Y_all[:, i+2] = y_apf[N_plot, :]
end
legend()

title("all")
subplot(2, 3, 2)
X_trig = zeros(length(Δ), 4)
Y_trig = zeros(length(Δ), 4)
for (i, k) in enumerate([likelihood_analytic, likelihood_MC])
    x_bpf, y_bpf, _, _, _, _ = extract_err_over_com(result, par_vec, "bpf", k, y_field=:ce_trig)
    x_apf, y_apf, _, _, _, _ = extract_err_over_com(result, par_vec, "apf", k, y_field=:ce_trig)
    plot(x_bpf[N_plot, :], y_bpf[N_plot, :], "C$(i-1)*", label="bpf")
    plot(x_apf[N_plot, :], y_apf[N_plot, :], "C$(i-1)s", label="apf")
    X_trig[:, i] = x_bpf[N_plot, :]
    X_trig[:, i+2] = x_apf[N_plot, :]
    Y_trig[:, i] = y_bpf[N_plot, :]
    Y_trig[:, i+2] = y_apf[N_plot, :]
end
title("γk = 1")
subplot(2, 3, 3)
X_noTrig = zeros(length(Δ), 4)
Y_noTrig = zeros(length(Δ), 4)
for (i, k) in enumerate([likelihood_analytic, likelihood_MC])
    x_bpf, y_bpf, _, _, _, _ = extract_err_over_com(result, par_vec, "bpf", k, y_field=:ce_noTrig)
    x_apf, y_apf, _, _, _, _ = extract_err_over_com(result, par_vec, "apf", k, y_field=:ce_noTrig)
    plot(x_bpf[N_plot, :], y_bpf[N_plot, :], "C$(i-1)*", label="bpf")
    plot(x_apf[N_plot, :], y_apf[N_plot, :], "C$(i-1)s", label="apf")
    X_noTrig[:, i] = x_bpf[N_plot, :]
    X_noTrig[:, i+2] = x_apf[N_plot, :]
    Y_noTrig[:, i] = y_bpf[N_plot, :]
    Y_noTrig[:, i+2] = y_apf[N_plot, :]
end
title("γk = 0, numint")
subplot(2, 3, 4)
Y_N_all = zeros(length(N_vec), 4)
for (i, k) in enumerate([likelihood_analytic, likelihood_MC])
    x_bpf, y_bpf, _, _, _, _ = extract_err_over_com(result, par_vec, "bpf", k, y_field=:ce_all)
    x_apf, y_apf, _, _, _, _ = extract_err_over_com(result, par_vec, "apf", k, y_field=:ce_all)
    plot(N_vec, y_bpf[:, δ_plot], "C$(i-1)*", label="bpf")
    plot(N_vec, y_apf[:, δ_plot], "C$(i-1)s", label="apf")
    Y_N_all[:, i] = y_bpf[:, δ_plot]
    Y_N_all[:, i+2] = y_apf[:, δ_plot]
end
subplot(2, 3, 5)
Y_N_trig = zeros(length(N_vec), 4)
for (i, k) in enumerate([likelihood_analytic, likelihood_MC])
    x_bpf, y_bpf, _, _, _, _ = extract_err_over_com(result, par_vec, "bpf", k, y_field=:ce_trig)
    x_apf, y_apf, _, _, _, _ = extract_err_over_com(result, par_vec, "apf", k, y_field=:ce_trig)
    plot(N_vec, y_bpf[:, δ_plot], "C$(i-1)*", label="bpf")
    plot(N_vec, y_apf[:, δ_plot], "C$(i-1)s", label="apf")
    Y_N_trig[:, i] = y_bpf[:, δ_plot]
    Y_N_trig[:, i+2] = y_apf[:, δ_plot]
end
subplot(2, 3, 6)
Y_N_noTrig = zeros(length(N_vec), 4)
for (i, k) in enumerate([likelihood_analytic, likelihood_MC])
    x_bpf, y_bpf, _, _, _, _ = extract_err_over_com(result, par_vec, "bpf", k, y_field=:ce_noTrig)
    x_apf, y_apf, _, _, _, _ = extract_err_over_com(result, par_vec, "apf", k, y_field=:ce_noTrig)
    plot(N_vec, y_bpf[:, δ_plot], "C$(i-1)*", label="bpf")
    plot(N_vec, y_apf[:, δ_plot], "C$(i-1)s", label="apf")
    Y_N_noTrig[:, i] = y_bpf[:, δ_plot]
    Y_N_noTrig[:, i+2] = y_apf[:, δ_plot]
end

## -------------------- SAVE DATA  ----------------------------

using CSV
using DataFrames



df_all = DataFrame( X_bpf = X_all[:, 1],
                    X_bpf_MC = X_all[:, 2],
                    X_apf = X_all[:, 3],
                    X_apf_MC = X_all[:, 4],
                    Y_bpf = Y_all[:, 1],
                    Y_bpf_MC = Y_all[:, 2],
                    Y_apf = Y_all[:, 3],
                    Y_apf_MC = Y_all[:, 4])

df_trig = DataFrame(X_bpf = X_trig[:, 1],
                    X_bpf_MC = X_trig[:, 2],
                    X_apf = X_trig[:, 3],
                    X_apf_MC = X_trig[:, 4],
                    Y_bpf = Y_trig[:, 1],
                    Y_bpf_MC = Y_trig[:, 2],
                    Y_apf = Y_trig[:, 3],
                    Y_apf_MC = Y_trig[:, 4])

df_noTrig = DataFrame(  X_bpf = X_noTrig[:, 1],
                        X_bpf_MC = X_noTrig[:, 2],
                        X_apf = X_noTrig[:, 3],
                        X_apf_MC = X_noTrig[:, 4],
                        Y_bpf = Y_noTrig[:, 1],
                        Y_bpf_MC = Y_noTrig[:, 2],
                        Y_apf = Y_noTrig[:, 3],
                        Y_apf_MC = Y_noTrig[:, 4])

df_numerr = DataFrame(  X_bpf = X_numerr[:, 1],
                        X_bpf_MC = X_numerr[:, 2],
                        X_apf = X_numerr[:, 3],
                        X_apf_MC = X_numerr[:, 4],
                        Y_bpf = Y_numerr[:, 1],
                        Y_bpf_MC = Y_numerr[:, 2],
                        Y_apf = Y_numerr[:, 3],
                        Y_apf_MC = Y_numerr[:, 4])

df_N_all = DataFrame(   X = N_vec,
                        Y_bpf = Y_N_all[:, 1],
                        Y_bpf_MC = Y_N_all[:, 2],
                        Y_apf = Y_N_all[:, 3],
                        Y_apf_MC = Y_N_all[:, 4])

df_N_trig = DataFrame(   X = N_vec,
                        Y_bpf = Y_N_trig[:, 1],
                        Y_bpf_MC = Y_N_trig[:, 2],
                        Y_apf = Y_N_trig[:, 3],
                        Y_apf_MC = Y_N_trig[:, 4])

df_N_noTrig = DataFrame(   X = N_vec,
                        Y_bpf = Y_N_noTrig[:, 1],
                        Y_bpf_MC = Y_N_noTrig[:, 2],
                        Y_apf = Y_N_noTrig[:, 3],
                        Y_apf_MC = Y_N_noTrig[:, 4])

CSV.write(savepath*"comparefiltertype_all.csv", df_all, writeheader=true)
CSV.write(savepath*"comparefiltertype_trig.csv", df_trig, writeheader=true)
CSV.write(savepath*"comparefiltertype_noTrig.csv", df_noTrig, writeheader=true)
CSV.write(savepath*"comparefiltertype_numerr.csv", df_numerr, writeheader=true)
CSV.write(savepath*"comparefiltertype_N_all.csv", df_N_all, writeheader=true)
CSV.write(savepath*"comparefiltertype_N_trig.csv", df_N_trig, writeheader=true)
CSV.write(savepath*"comparefiltertype_N_noTrig.csv", df_N_noTrig, writeheader=true)

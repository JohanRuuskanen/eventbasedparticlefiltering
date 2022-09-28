
# Estimate the triggering probability using the particle filter and compare with
# an Monte Carlo estimate of the triggering probability.

using JLD
using Random
using Distributions
using LinearAlgebra
using EventBasedParticleFiltering
using PyPlot
using StatsBase

const EP = EventBasedParticleFiltering

savepath = "/path/to/save" # Change this

## -------------------- DECLARE COMMON VARIABLES ----------------------------

@everywhere begin

    using Random
    using StatsBase
    using Distributions
    using LinearAlgebra
    using EventBasedParticleFiltering

    const EP = EventBasedParticleFiltering

    # Reduce the parameter sweep if running on a single computer

    MCsims = 300
    sims = 10000

    D = 3
    N_vec = [25, 50, 100, 200, 400]
    LH_evals = [likelihood_analytic(), likelihood_MC(M=1)]
    T = 100
    Δ = [2.5, 7.5]
    system = "nonlinear_classic"

    sys, _, _, _, _, p0 = EP.generate_example_system(T, type=system)
    q, qv = EP.genprop_linear_gaussian_noise()

end

@everywhere function simfunc(meta, params)
    println("Running $(meta[1]) / $(meta[2])")

    Random.seed!(meta[1]*7319 + 8192741)

    if params["filtertype"] == "bpf"
        opt = ebpf_options(sys=sys, N=params["N"], kernel=kernel_IBT([params["δ"]]),
            pftype=pftype_bootstrap(),
            likelihood=params["lh"],
            triggerat="never",
            print_progress=false)
    elseif params["filtertype"] == "apf"
        opt = ebpf_options(sys=sys, N=params["N"], kernel=kernel_IBT([params["δ"]]),
            pftype=pftype_auxiliary(qv=qv, q=q, D=D),
            likelihood=params["lh"],
            triggerat="never",
            print_progress=false)
    end

    # Calculate p_T using the particle filter
    x, y = sim_sys(sys, x0=rand(p0))
    output = ebpf(y, opt, X0=rand(p0, params["N"])')

    p_first = zeros(T)
    for k = 1:T
        p_first[k] = output.p_trig[k]
        for i = 1:(k-1)
            p_first[k] *= (1 - output.p_trig[k-i])
        end
    end

    # Calculate p_T using monte carlo
    opt.triggerat = "events"
    opt.abort_at_trig = true

    vals = zeros(sims)
    for sim = 1:sims
        x, y = sim_sys(sys, x0=rand(p0))
        output = ebpf(y, opt, X0=rand(p0, params["N"])')
        Γ = output.Γ
        Γ[1] = 0
        v = findfirst(x -> x==1, Γ)
        if v == nothing
            vals[sim] = T
        else
            vals[sim] = v
        end
    end

    u = countmap(vals)
    occ = [if haskey(u, i); u[i]; else; 0; end for i = 1:T]
    e_first = occ ./sims

    return p_first, e_first
end

## -------------------- RUN DISTRIBUTED COMPUTING ----------------------------
printstyled("Running distributed task on all workers\n", bold=true, color=:magenta)

Random.seed!(5)

p_first = zeros(T, MCsims, 2)
steps = zeros(sims, MCsims, 2)

par_vec = Array{Dict, 1}(undef, 0)
for N in N_vec
    global par_vec
    for δ in Δ
        for lh in LH_evals
            for mcsim = 1:MCsims
                par_vec = vcat(par_vec, Dict("filtertype" => "bpf",
                                                "N" => N,
                                                "δ" => δ,
                                                "lh" => lh,
                                                "mcsim" => mcsim))
                par_vec = vcat(par_vec, Dict("filtertype" => "apf",
                                                "N" => N,
                                                "δ" => δ,
                                                "lh" => lh,
                                                "mcsim" => mcsim))
            end
        end
    end
end

results = pmap(1:length(par_vec), retry_delays=exp.(1:5)) do index
    result = simfunc((index, length(par_vec)), par_vec[index])
end

printstyled("All tasks complete!\n", bold=true, color=:magenta)

save(savepath*"triggerestimation.jld",
        "par_vec", par_vec,
        "result", results)


## -------------------- PLOT RESULTS ----------------------------

result = load(savepath*"triggerestimation.jld", "result")
par_vec = load(savepath*"triggerestimation.jld", "par_vec")

MCsims = 300
sims = 10000
filtertypes = ["bpf", "apf"]
D = 3
N_vec = [25, 50, 100, 200, 400]
LH_evals = [likelihood_analytic(), likelihood_MC(M=1)]
T = 100
Δ = [2.5, 7.5]

function Tc_itr(T, p_T, c)

    function ΔTc(n)
        cnU = floor(Int64, c*(n+1))
        cnL = floor(Int64, c*n)

        a = (c*(n + 1) - cnU)*(cnU - cnL)
        if a > 0
            a *= p_T[cnU]
        end

        return a + 1 - sum(p_T[1:n]) - c*(1 - sum(p_T[1:cnL]))

    end

    Tc = zeros(T)
    Tc[1] = ΔTc(0)

    for k = 2:T
        Tc[k] = Tc[k-1] + ΔTc(k-1)
    end

    return Tc
end

function find_quant(p_T, a)
    n = 1
    while sum(p_T[1:n]) < a
        n += 1
    end
    return n
end

function plot_E(p_T, c)
    for k = 1:length(c)
        Tc = Tc_itr(T, p_T, c[k])
        nsh = find_quant(p_T, 1-c[k])
        nopt = argmax(Tc)
        plot(x, Tc, "C$(k-1)")
        plot(nsh, Tc[nsh], "C$(k-1)o", markeredgewidth=1.5, markeredgecolor="k")
        plot(nopt, Tc[nopt], "C$(k-1)^", markeredgewidth=1.5, markeredgecolor="k")
    end
end

function plot_E_pf(p_T, c; style="")
    for k = 1:length(c)
        Tc = Tc_itr(T, p_T, c[k])
        nh = findfirst(x -> x < 0, diff(Tc))
        plot(x, Tc, "C$(k-1)"*style)
        plot(nh, Tc[nh], "C$(k-1)s", markeredgewidth=1.5, markeredgecolor="k")
    end
end

p_err = zeros(length(filtertypes), length(N_vec), length(Δ),
                length(LH_evals), MCsims)
p_trig_all = zeros(length(filtertypes), length(N_vec), length(Δ),
                length(LH_evals), MCsims, T)
e_trig_all = zeros(length(filtertypes), length(N_vec), length(Δ),
                length(LH_evals), MCsims, T)

for (i1, ft) in enumerate(filtertypes)
    for (i2, N) in enumerate(N_vec)
        for (i3, δ) in enumerate(Δ)
            for (i4, lh) in enumerate(LH_evals)
                for i5 = 1:MCsims
                    for k = 1:length(par_vec)
                        if par_vec[k]["filtertype"] == ft &&
                                par_vec[k]["N"] == N && par_vec[k]["δ"] == δ &&
                                typeof(par_vec[k]["lh"]) == typeof(lh) &&
                                par_vec[k]["mcsim"] == i5

                            p_T = result[k][1]
                            e_T = result[k][2]
                            p_err[i1, i2, i3, i4, i5] = sqrt(mean((p_T - e_T).^2))
                            p_trig_all[i1, i2, i3, i4, i5, :] = p_T
                            e_trig_all[i1, i2, i3, i4, i5, :] = e_T
                        end
                    end
                end
            end
        end
    end
end

function extract_statistics(err_sims)
    ql = zeros(size(err_sims, 1))
    qu = zeros(size(err_sims, 1))
    for k = 1:size(err_sims, 1)
        ql[k], qu[k] = quantile(err_sims[k, :], [0.05, 0.95])
    end
    m = mean(err_sims, dims=2)
    return m, ql, qu
end

function extract_statistics_ts(err_sims)
    ql = zeros(size(err_sims, 2))
    qu = zeros(size(err_sims, 2))
    for k = 1:size(err_sims, 2)
        ql[k], qu[k] = quantile(err_sims[:, k], [0.05, 0.95])
    end
    m = mean(err_sims, dims=1)
    return m[:], ql[:], qu[:]
end

figure(1)
clf()
m_bpf_1, ql_bpf_1, qu_bpf_1 = extract_statistics(p_err[1, :, 1, 2, :])
m_apf_1, ql_apf_1, qu_apf_1 = extract_statistics(p_err[2, :, 1, 2, :])
plot(N_vec, m_bpf_1, "C0*-", label="bpf Δ = 2.5")
plot(N_vec, m_apf_1, "C0*--", label="apf Δ = 2.5")
m_bpf_2, ql_bpf_2, qu_bpf_2 = extract_statistics(p_err[1, :, 2, 2, :])
m_apf_2, ql_apf_2, qu_apf_2 = extract_statistics(p_err[2, :, 2, 2, :])
plot(N_vec, m_bpf_2, "C1*-", label="bpf Δ = 7.5")
plot(N_vec, m_apf_2, "C1*--", label="bpf Δ = 7.5")
ylabel("error")
xlabel("N")
legend()

t_end = 40
T_p = 0:t_end-1
figure(2)
clf()
m_mc, ql_mc, qu_mc = extract_statistics_ts(e_trig_all[2, 5, 2, 1, :, :])
m_apf, ql_apf, qu_apf = extract_statistics_ts(p_trig_all[2, 5, 2, 2, :, :])
step(T_p, m_mc[1:t_end], where="mid", color="C0", label="MC mean")
step(T_p, m_apf[1:t_end], where="mid", color="C1", label="PF mean")
fill_between(T_p, ql_mc[1:t_end], qu_mc[1:t_end], step="mid", alpha=0.25, label="MC int")
fill_between(T_p, ql_apf[1:t_end], qu_apf[1:t_end], step="mid", alpha=0.25, label="PF int")
ylabel("p(n)")
xlabel("n")
legend()



## Save data

using CSV
using DataFrames

df_erroverN = DataFrame( X = N_vec,
                    Y_bpf_2 = m_bpf_1[:],
                    Y_bpf_7 = m_bpf_2[:],
                    Y_apf_2 = m_apf_1[:],
                    Y_apf_7 = m_apf_2[:])

df_trigprob = DataFrame(X = T_p,
                        Y_mc = m_mc[1:t_end],
                        Y_mc_L = ql_mc[1:t_end],
                        Y_mc_U = qu_mc[1:t_end],
                        Y_apf = m_apf[1:t_end],
                        Y_apf_L = ql_apf[1:t_end],
                        Y_apf_U = qu_apf[1:t_end])

CSV.write(savepath*"triggerest_erroverN.csv", df_erroverN, writeheader=true)
CSV.write(savepath*"triggerest_trigprob.csv", df_trigprob, writeheader=true)


using Random
using PyPlot
using StatsBase
using Distributions
using LinearAlgebra
using EventBasedParticleFiltering

const EP = EventBasedParticleFiltering

## calculate the triggering probabilities

function calc_trig_probs(opt::ebpf_options)
    x, y = sim_sys(sys, x0=rand(p0))
    # Via PF
    println("")
    output = ebpf(y, opt, X0=rand(p0, N)')
    p_first = zeros(T)
    for k = 1:T
        p_first[k] = output.p_trig[k]
        for i = 1:(k-1)
            p_first[k] *= (1 - output.p_trig[k-i])
        end
    end

    # Via naive MC
    println("")
    MC_trig = zeros(MC)
    for k = 1:MC
        print("Running $(k) / $(MC) \r")
        flush(stdout)
        x, y = sim_sys(sys, x0=rand(p0))

        opt.abort_at_trig = true
        opt.print_progress = false
        opt.triggerat = "events"

        output_sim = ebpf(y, opt, X0=rand(p0, N)')
        Γ = output_sim.Γ
        Γ[1] = 0
        val = findfirst(x -> x==1, Γ)
        if val == nothing
            val = T
        end
        MC_trig[k] = val
    end

    u = countmap(MC_trig)
    val = [i for i = 1:T]
    occ = [if haskey(u, i); u[i]; else; 0; end for i = 1:T]
    e_first = occ ./ MC

    return p_first, e_first
end

Random.seed!(123)

MC = 1000
N = 400
T = 100

lh = likelihood_MC(M=1)

sys, A, C, Q, R, p0 = EP.generate_example_system(T, type="nonlinear_classic")
q, qv = EP.genprop_linear_gaussian_noise()

opt1 = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([2.5]),
    pftype=pftype_auxiliary(qv=qv, q=q, D=5), triggerat="never", likelihood=lh)
opt2 = ebpf_options(sys=sys, N=N, kernel=kernel_IBT([7.5]),
    pftype=pftype_auxiliary(qv=qv, q=q, D=5), triggerat="never", likelihood=lh)

p_first1, e_first1 = calc_trig_probs(opt1)
p_first2, e_first2 = calc_trig_probs(opt2)

## Calculate and display the cost based on the estimated p_T

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

c = [0.05, 0.1, 0.25]
x = 1:T
figure(1)
clf()
subplot(2, 1, 1)
plot_E(e_first1, c)
plot_E_pf(p_first1, c, style="--")
subplot(2, 1, 2)
plot_E(e_first2, c)
plot_E_pf(p_first2, c, style="--")
title("")
## -------------------- SAVE DATA  ----------------------------

savepath = "/path/to/save" # Change this

using CSV
using DataFrames

nh_v = repeat(Array{Any,1}(["nan"]), T, length(c), 2)
nsh_v = repeat(Array{Any,1}(["nan"]), T, length(c), 2)
nopt_v = repeat(Array{Any,1}(["nan"]), T, length(c), 2)
Tc = zeros(T, length(c), 2, 2)

probs = [(e_first1, p_first1), (e_first2, p_first2)]

for (i, prob) in enumerate(probs)
    for k = 1:length(c)
        Tc[:, k, 1, i] = Tc_itr(T, prob[1], c[k])
        Tc[:, k, 2, i] = Tc_itr(T, prob[2], c[k])

        nh = findfirst(x -> x < 0, diff(Tc[:, k, 2, i]))
        nh_v[nh, k, i] = Tc[nh, k, 2, i]

        nsh = find_quant(prob[1], 1-c[k])
        nsh_v[nsh, k, i] = Tc[nsh, k, 1, i]

        nopt = argmax(Tc[:, k, 1, i])
        nopt_v[nopt, k, i] = Tc[nopt, k, 1, i]

    end
end

i = 1
df1 = DataFrame(   X = collect(1:T),
                    Tc_1 = Tc[:, 1, 1, i],
                    Tc_1_nsh = nsh_v[:, 1, i],
                    Tc_1_nopt = nopt_v[:, 1, i],
                    Tc_1_apf = Tc[:, 1, 2, i],
                    Tc_1_nh_apf = nh_v[:, 1, i],
                    Tc_2 = Tc[:, 2, 1, i],
                    Tc_2_nsh = nsh_v[:, 2, i],
                    Tc_2_nopt = nopt_v[:, 2, i],
                    Tc_2_apf = Tc[:, 2, 2, i],
                    Tc_2_nh_apf = nh_v[:, 2, i],
                    Tc_3 = Tc[:, 3, 1, i],
                    Tc_3_nsh = nsh_v[:, 3, i],
                    Tc_3_nopt = nopt_v[:, 3, i],
                    Tc_3_apf = Tc[:, 3, 2, i],
                    Tc_3_nh_apf = nh_v[:, 3, i])

i = 2
df2 = DataFrame(   X = collect(1:T),
                    Tc_1 = Tc[:, 1, 1, i],
                    Tc_1_nsh = nsh_v[:, 1, i],
                    Tc_1_nopt = nopt_v[:, 1, i],
                    Tc_1_apf = Tc[:, 1, 2, i],
                    Tc_1_nh_apf = nh_v[:, 1, i],
                    Tc_2 = Tc[:, 2, 1, i],
                    Tc_2_nsh = nsh_v[:, 2, i],
                    Tc_2_nopt = nopt_v[:, 2, i],
                    Tc_2_apf = Tc[:, 2, 2, i],
                    Tc_2_nh_apf = nh_v[:, 2, i],
                    Tc_3 = Tc[:, 3, 1, i],
                    Tc_3_nsh = nsh_v[:, 3, i],
                    Tc_3_nopt = nopt_v[:, 3, i],
                    Tc_3_apf = Tc[:, 3, 2, i],
                    Tc_3_nh_apf = nh_v[:, 3, i])

CSV.write(savepath*"TC_delta1.csv", df1, writeheader=true)
CSV.write(savepath*"TC_delta2.csv", df2, writeheader=true)

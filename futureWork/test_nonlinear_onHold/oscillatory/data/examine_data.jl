ENV["JULIA_PKGDIR"] = "/local/home/johanr/.julia"

using JLD
using PyPlot
include("/local/home/johanr/Projects/EBPF/src/misc.jl")
include("/local/home/johanr/Projects/EBPF/src/pyplot2tikz.jl")

read_new = true
#N = [100 250 500 1000]
#Δ = [5]
#path = "/local/home/johanr/Projects/EBPF/test_nonlinear/data/test_run_over_N_small_delta5"

#N = [250]
#Δ = [0 1 2 3 4 5 6 7 8 9 10]
#path = "/local/home/johanr/Projects/EBPF/test_nonlinear/data/test_run_over_D"

N = [250]
Δ = [0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0]
path = "/local/home/johanr/Projects/EBPF/test_nonlinear/oscillatory/data/test_run_over_smallD"

m = length(N)
n = length(Δ)
k = 1000

bpf_var = "bpf2"
apf_var = "apf2"

function calc_recursive(M, n, y)
    T = size(y, 2)
    for i = 1:T
        tmp = M

        M += (y[:, i] - M) / (n + 1)
        n += 1
    end
    return M, n
end

if read_new
    Err_bpf = Dict{String,Array{Float64}}(
        "mean" => zeros(m, n, 1),
        "mean_g1" => zeros(m, n, 1),
        "Neff" => zeros(m, n),
        "fail" => zeros(m, n),
        "res" => zeros(m, n),
        "trig" => zeros(m, n)
    )
    Err_apf = Dict{String,Array{Float64}}(
        "mean" => zeros(m, n, 1),
        "mean_g1" => zeros(m, n, 1),
        "Neff" => zeros(m, n),
        "fail" => zeros(m, n),
        "res" => zeros(m, n),
        "trig" => zeros(m, n)
    )

    for i1 = 1:m
        for i2 = 1:n
            println(string(i1)*" "*string(i2))

            E = load(path * "/sim_" * string(i1) * "_" * string(i2) * ".jld", "results")

            counters = Dict{String,Int64}(
                "ebpf" => 0,
                "eapf" => 0,
                "ebpf_g1" => 0,
                "eapf_g1" => 0,
            )



            for i3 = 1:k
                bpf_measure = E[i3]["err_"*bpf_var]
                apf_measure = E[i3]["err_"*apf_var]

                Γ_bpf = E[i3]["trig_"*bpf_var]
                Γ_apf = E[i3]["trig_"*apf_var]

                Err_bpf["Neff"][i1, i2] += sum(E[i3]["Neff_"*bpf_var])
                Err_apf["Neff"][i1, i2] += sum(E[i3]["Neff_"*apf_var])

                Err_bpf["fail"][i1, i2] += 0#sum(E[i3]["fail_"*bpf_var])
                Err_apf["fail"][i1, i2] += 0#sum(E[i3]["fail_"*apf_var])

                Err_bpf["res"][i1, i2] += sum(E[i3]["res_"*bpf_var])
                Err_apf["res"][i1, i2] += sum(E[i3]["res_"*apf_var])


                Err_bpf["trig"][i1, i2] += sum(E[i3]["trig_"*bpf_var])
                Err_apf["trig"][i1, i2] += sum(E[i3]["trig_"*apf_var])


                if length(bpf_measure) - length(apf_measure) == 0
                    T = size(bpf_measure, 2)
                else
                    println("Warning: lengths not equal!")
                end

                idx_bpf = find(x -> x == 1, Γ_bpf)
                idx_apf = find(x -> x == 1, Γ_apf)

                Err_bpf["mean"][i1, i2, :], counters["ebpf"] =
                    calc_recursive(Err_bpf["mean"][i1, i2, :], counters["ebpf"],
                                    bpf_measure.^2)
                Err_apf["mean"][i1, i2, :], counters["eapf"] =
                    calc_recursive(Err_apf["mean"][i1, i2, :], counters["eapf"],
                                    apf_measure.^2)

                Err_bpf["mean_g1"][i1, i2, :], counters["ebpf_g1"] =
                    calc_recursive(Err_bpf["mean_g1"][i1, i2, :], counters["ebpf_g1"],
                                    bpf_measure[:, idx_bpf].^2)
                Err_apf["mean_g1"][i1, i2, :], counters["eapf_g1"] =
                    calc_recursive(Err_apf["mean_g1"][i1, i2, :], counters["eapf_g1"],
                                    apf_measure[:, idx_apf].^2)
            end

            Err_bpf["Neff"][i1, i2] /= 1000 * k
            Err_apf["Neff"][i1, i2] /= 1000 * k

            Err_bpf["fail"][i1, i2] /= 1000
            Err_apf["fail"][i1, i2] /= 1000

            Err_bpf["res"][i1, i2] /= 1000
            Err_apf["res"][i1, i2] /= 1000

            Err_bpf["trig"][i1, i2] /= 1000
            Err_apf["trig"][i1, i2] /= 1000

        end
    end
end

N_lags = 1
Δ_lags = 1

fig1 = figure(1)
clf()
title("State x_1")
m = [Err_bpf["mean"][N_lags, :, 1], Err_apf["mean"][N_lags, :, 1]]
plot(Δ[:], m[1], "C1o-", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
plot(Δ[:], m[2], "C3s-", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
legend(["BPF", "APF", "EBSE", "Kalman"])
grid(true)

fig2 = figure(2)
clf()
title("State x_1")
m = [Err_bpf["mean_g1"][N_lags, :, 1], Err_apf["mean_g1"][N_lags, :, 1]]
plot(Δ[:], m[1], "C1o-", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
plot(Δ[:], m[2], "C3s-", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
legend(["BPF", "APF", "EBSE", "Kalman"])
grid(true)

fig3 = figure(3)
clf()
subplot(4, 1, 1)
title("Effective sample size")
plot(Δ[:], Err_bpf["Neff"][N_lags, :], "C0x-")
plot(Δ[:], Err_apf["Neff"][N_lags, :], "C1o-")
legend(["BPF", "APF"])
subplot(4, 1, 2)
title("Number of failures")
plot(Δ[:], Err_bpf["fail"][N_lags, :], "C0x-")
plot(Δ[:], Err_apf["fail"][N_lags, :], "C1o-")
legend(["BPF", "APF"])
subplot(4, 1, 3)
title("Resamples")
plot(Δ[:], Err_bpf["res"][N_lags, :], "C0x-")
plot(Δ[:], Err_apf["res"][N_lags, :], "C1o-")
legend(["BPF", "APF"])
subplot(4, 1, 4)
title("Measurement values")
plot(Δ[:], Err_bpf["trig"][N_lags, :], "C0x-")
plot(Δ[:], Err_apf["trig"][N_lags, :], "C1o-")
legend(["BPF", "APF"])

#=
fig3 = figure(4)
clf()
title("All values, State x_1")
m = [log.(Err_bpf["mean"][:, Δ_lags, 1]), log.(Err_apf["mean"][:, Δ_lags, 1])]
plot(N[:], m[1], "C1o-", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
plot(N[:], m[2], "C3s-", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
legend(["BPF", "APF", "EBSE", "Kalman"])

fig4 = figure(5)
clf()
title("Measurement values, State x_1")
m = [log.(Err_bpf["mean_g1"][:, Δ_lags, 1]), log.(Err_apf["mean_g1"][:, Δ_lags, 1])]
plot(N[:], m[1], "C1o-", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
plot(N[:], m[2], "C3s-", markeredgewidth=1.5, markeredgecolor=(0,0,0,1))
legend(["BPF", "APF", "EBSE", "Kalman"])

fig5 = figure(6)
clf()
subplot(4, 1, 1)
title("Effective sample size")
plot(N[:], Err_bpf["Neff"][:, Δ_lags], "C0x-")
plot(N[:], Err_apf["Neff"][:, Δ_lags], "C1o-")
legend(["BPF", "APF"])
subplot(4, 1, 2)
title("Number of failures")
plot(N[:], Err_bpf["fail"][:, Δ_lags], "C0x-")
plot(N[:], Err_apf["fail"][:, Δ_lags], "C1o-")
legend(["BPF", "APF"])
subplot(4, 1, 3)
title("Triggered resamples")
plot(N[:], Err_bpf["res"][:, Δ_lags], "C0x-")
plot(N[:], Err_apf["res"][:, Δ_lags], "C1o-")
legend(["BPF", "APF"])
subplot(4, 1, 4)
title("Measurement values")
plot(N[:], Err_bpf["trig"][:, Δ_lags], "C0x-")
plot(N[:], Err_apf["trig"][:, Δ_lags], "C1o-")
legend(["BPF", "APF"])
=#

ENV["JULIA_PKGDIR"] = "/home/johanr/.julia"

using JLD
using PyPlot

read_new = false

N = [10 25 50 75 100 150 200 250 300 350 400 500]
Δ = [0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6 4.0]

m = length(N)
n = length(Δ)
L = 50
k = 1000

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
        "mean" => zeros(m, n, 2),
        "mean_g1" => zeros(m, n, 2),
        "Neff" => zeros(m, n),
        "fail" => zeros(m, n),
        "res" => zeros(m, n)
    )
    Err_apf = Dict{String,Array{Float64}}(
        "mean" => zeros(m, n, 2),
        "mean_g1" => zeros(m, n, 2),
        "Neff" => zeros(m, n),
        "fail" => zeros(m, n),
        "res" => zeros(m, n)
    )
    Err_ebse = Dict{String,Array{Float64}}(
        "mean" => zeros(n, 2),
        "mean_g1" => zeros(n, 2)
    )

    Err_kalman = Dict{String,Array{Float64}}(
        "mean" => zeros(2, 1)
    )

    for i1 = 1:m
        for i2 = 1:n
            println(string(i1)*" "*string(i2))

            E = load("/home/johanr/projects/EBPF/test_linear/data/test_linear_system_run3/sim_" * string(i1) * "_" * string(i2) * ".jld", "results")

            counters = Dict{String,Int64}(
                "ebpf" => 0,
                "eapf" => 0,
                "ebse" => 0,
                "ebpf_g1" => 0,
                "eapf_g1" => 0,
                "ebse_g1" => 0,
                "kalman" => 0
            )

            for i3 = 1:k
                bpf_measure = E[i3]["err_ebpf"]
                apf_measure = E[i3]["err_eapf"]

                Γ_bpf = E[i3]["trig_ebpf"]
                Γ_apf = E[i3]["trig_eapf"]

                Err_bpf["Neff"][i1, i2] += sum(E[i3]["Neff_ebpf"])
                Err_apf["Neff"][i1, i2] += sum(E[i3]["Neff_eapf"])

                Err_bpf["fail"][i1, i2] += sum(E[i3]["fail_ebpf"])
                Err_apf["fail"][i1, i2] += sum(E[i3]["fail_eapf"])

                Err_bpf["res"][i1, i2] += sum(E[i3]["res_ebpf"])
                Err_apf["res"][i1, i2] += sum(E[i3]["res_eapf"])


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

                if i1 == 1
                    if i2 == 1
                        Err_kalman["mean"], counters["kalman"] =
                            calc_recursive(Err_kalman["mean"], counters["kalman"],
                                            E[i3]["err_kal"].^2)
                    end

                    ebse_measure = E[i3]["err_ebse"]
                    Γ_ebse = E[i3]["trig_ebse"]

                    idx_ebse = find(x -> x == 1, Γ_ebse)

                    Err_ebse["mean"][i2, :], counters["ebse"] =
                        calc_recursive(Err_ebse["mean"][i2, :], counters["ebse"],
                                        ebse_measure.^2)

                    Err_ebse["mean_g1"][i2, :], counters["ebse_g1"] =
                        calc_recursive(Err_ebse["mean_g1"][i2, :], counters["ebse_g1"],
                                        ebse_measure[:, idx_ebse].^2)

                end
            end

            Err_bpf["Neff"][i1, i2] /= 1000 * k
            Err_apf["Neff"][i1, i2] /= 1000 * k

            Err_bpf["fail"][i1, i2] /= 1000
            Err_apf["fail"][i1, i2] /= 1000

            Err_bpf["res"][i1, i2] /= 1000
            Err_apf["res"][i1, i2] /= 1000

        end
    end
end


N_lags = 12
Δ_lags = 11

figure(1)
clf()
subplot(1, 2, 1)
title("State x_1")
m = [Err_bpf["mean"][N_lags, :, 1],
     Err_apf["mean"][N_lags, :, 1],
     Err_ebse["mean"][:, 1],
     Err_kalman["mean"][1]]
plot(Δ[:], m[1], "C0x-")
plot(Δ[:], m[2], "C1o-")
plot(Δ[:], m[3], "C2^-")
plot(Δ[:], m[4]*ones(n), "r--")
legend(["BPF", "APF", "EBSE", "Kalman"])
subplot(1, 2, 2)
title("State x_2")
m = [Err_bpf["mean"][N_lags, :, 2],
     Err_apf["mean"][N_lags, :, 2],
     Err_ebse["mean"][:, 2],
     Err_kalman["mean"][2]]
plot(Δ[:], m[1], "C0x-")
plot(Δ[:], m[2], "C1o-")
plot(Δ[:], m[3], "C2^-")
plot(Δ[:], m[4]*ones(n), "r--")

figure(2)
clf()
subplot(1, 2, 1)
title("State x_1")
m = [Err_bpf["mean_g1"][N_lags, :, 1],
     Err_apf["mean_g1"][N_lags, :, 1],
     Err_ebse["mean_g1"][:, 1],
     Err_kalman["mean"][1]]
plot(Δ[:], m[1], "C0x-")
plot(Δ[:], m[2], "C1o-")
plot(Δ[:], m[3], "C2^-")
plot(Δ[:], m[4]*ones(n), "r--")
legend(["BPF", "APF", "EBSE", "Kalman"])
subplot(1, 2, 2)
title("State x_2")
m = [Err_bpf["mean_g1"][N_lags, :, 2],
     Err_apf["mean_g1"][N_lags, :, 2],
     Err_ebse["mean_g1"][:, 2],
     Err_kalman["mean"][2]]
plot(Δ[:], m[1], "C0x-")
plot(Δ[:], m[2], "C1o-")
plot(Δ[:], m[3], "C2^-")
plot(Δ[:], m[4]*ones(n), "r--")

figure(3)
clf()
subplot(1, 2, 1)
title("State x_1")
m = [log.(Err_bpf["mean_g1"][:, Δ_lags, 1]),
     log.(Err_apf["mean_g1"][:, Δ_lags, 1]),
     log.(Err_ebse["mean_g1"][Δ_lags, 1]),
     log.(Err_kalman["mean"][1])]
plot(N[:], m[1], "C0x-")
plot(N[:], m[2], "C1o-")
plot(N[:], m[3]*ones(length(N)), "C2^-")
plot(N[:], m[4]*ones(length(N)), "r--")
legend(["BPF", "APF", "EBSE", "Kalman"])
subplot(1, 2, 2)
title("State x_2")
m = [log.(Err_bpf["mean_g1"][:, Δ_lags, 2]),
     log.(Err_apf["mean_g1"][:, Δ_lags, 2]),
     log.(Err_ebse["mean_g1"][Δ_lags, 2]),
     log.(Err_kalman["mean"][2])]
plot(N[:], m[1], "C0x-")
plot(N[:], m[2], "C1o-")
plot(N[:], m[3]*ones(length(N)), "C2^-")
plot(N[:], m[4]*ones(length(N)), "r--")


figure(4)
clf()
subplot(3, 1, 1)
title("Effective sample size")
plot(Δ[:], Err_bpf["Neff"][N_lags, :], "C0x-")
plot(Δ[:], Err_apf["Neff"][N_lags, :], "C1o-")
subplot(3, 1, 2)
title("Number of failures")
plot(Δ[:], Err_bpf["fail"][N_lags, :], "C0x-")
plot(Δ[:], Err_apf["fail"][N_lags, :], "C1o-")
subplot(3, 1, 3)
title("Triggered resamples")
plot(Δ[:], Err_bpf["res"][N_lags, :], "C0x-")
plot(Δ[:], Err_apf["res"][N_lags, :], "C1o-")

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

#TODO: IMPLEMENT RECURSIVE UPDATE TO MEAN AND VARIANCE

function calc_recursive(M, V, n, y)
    T = size(y, 2)
    for i = 1:T
        tmp = M

        M += (y[:, i] - M) / (n + 1)
        V += tmp.^2 - M.^2 + (y[:, i].^2 - V - tmp.^2)/(n+1)

        n += 1
    end
    return M, V, n
end

if read_new
    Err_bpf = Dict{String,Array{Float64}}(
        "mean" => zeros(m, n, 2),
        "var" => zeros(m, n, 2),
        "mean_g1" => zeros(m, n, 2),
        "var_g1" => zeros(m, n, 2)
    )
    Err_sis = Dict{String,Array{Float64}}(
        "mean" => zeros(m, n, 2),
        "var" => zeros(m, n, 2),
        "mean_g1" => zeros(m, n, 2),
        "var_g1" => zeros(m, n, 2)
    )
    Err_apf = Dict{String,Array{Float64}}(
        "mean" => zeros(m, n, 2),
        "var" => zeros(m, n, 2),
        "mean_g1" => zeros(m, n, 2),
        "var_g1" => zeros(m, n, 2)
    )
    Err_ebse = Dict{String,Array{Float64}}(
        "mean" => zeros(n, 2),
        "var" => zeros(n, 2),
        "mean_g1" => zeros(n, 2),
        "var_g1" => zeros(n, 2)
    )

    Err_kalman = Dict{String,Array{Float64}}(
        "mean" => zeros(2, 1),
        "var" => zeros(2, 1)
    )

    for i1 = 1:m
        for i2 = 1:n
            println(string(i1)*" "*string(i2))

            E = load("/home/johanr/projects/EBPF/test_linear/data/test_linear_system_run2/sim_" * string(i1) * "_" * string(i2) * ".jld", "results")

            counters = Dict{String,Int64}(
                "ebpf" => 0,
                "esis" => 0,
                "eapf" => 0,
                "ebse" => 0,
                "ebpf_g1" => 0,
                "esis_g1" => 0,
                "eapf_g1" => 0,
                "ebse_g1" => 0,
                "kalman" => 0,
            )

            for i3 = 1:k
                bpf_measure = E[i3]["err_ebpf"]
                sis_measure = E[i3]["err_esis"]
                apf_measure = E[i3]["err_eapf"]

                Γ_bpf = E[i3]["trig_ebpf"]
                Γ_sis = E[i3]["trig_esis"]
                Γ_apf = E[i3]["trig_eapf"]

                if 2*length(bpf_measure) - length(sis_measure) - length(apf_measure) == 0
                    T = size(bpf_measure, 2)
                else
                    println("Warning: lengths not equal!")
                end

                idx_bpf = find(x -> x == 1, Γ_bpf)
                idx_sis = find(x -> x == 1, Γ_sis)
                idx_apf = find(x -> x == 1, Γ_apf)

                Err_bpf["mean"][i1, i2, :], Err_bpf["var"][i1, i2, :], counters["ebpf"] =
                    calc_recursive(Err_bpf["mean"][i1, i2, :], Err_bpf["var"][i1, i2, :],
                                    counters["ebpf"], bpf_measure.^2)
                Err_sis["mean"][i1, i2, :], Err_sis["var"][i1, i2, :], counters["esis"] =
                    calc_recursive(Err_sis["mean"][i1, i2, :], Err_sis["var"][i1, i2, :],
                                    counters["esis"], sis_measure.^2)
                Err_apf["mean"][i1, i2, :], Err_apf["var"][i1, i2, :], counters["eapf"] =
                    calc_recursive(Err_apf["mean"][i1, i2, :], Err_apf["var"][i1, i2, :],
                                    counters["eapf"], apf_measure.^2)

                Err_bpf["mean_g1"][i1, i2, :], Err_bpf["var_g1"][i1, i2, :], counters["ebpf_g1"] =
                    calc_recursive(Err_bpf["mean_g1"][i1, i2, :], Err_bpf["var_g1"][i1, i2, :],
                                    counters["ebpf_g1"], bpf_measure[:, idx_bpf].^2)
                Err_sis["mean_g1"][i1, i2, :], Err_sis["var_g1"][i1, i2, :], counters["esis_g1"] =
                    calc_recursive(Err_sis["mean_g1"][i1, i2, :], Err_sis["var_g1"][i1, i2, :],
                                    counters["esis_g1"], sis_measure[:, idx_sis].^2)
                Err_apf["mean_g1"][i1, i2, :], Err_apf["var_g1"][i1, i2, :], counters["eapf_g1"] =
                    calc_recursive(Err_apf["mean_g1"][i1, i2, :], Err_apf["var_g1"][i1, i2, :],
                                    counters["eapf_g1"], apf_measure[:, idx_apf].^2)

                if i1 == 1
                    if i2 == 1
                        Err_kalman["mean"], Err_kalman["var"], counters["kalman"] =
                            calc_recursive(Err_kalman["mean"], Err_kalman["var"],
                                            counters["kalman"], E[i3]["err_kal"].^2)
                    end

                    ebse_measure = E[i3]["err_ebse"]
                    Γ_ebse = E[i3]["trig_ebse"]

                    idx_ebse = find(x -> x == 1, Γ_ebse)

                    Err_ebse["mean"][i2, :], Err_ebse["var"][i2, :], counters["ebse"] =
                        calc_recursive(Err_ebse["mean"][i2, :], Err_ebse["var"][i2, :],
                                        counters["ebse"], ebse_measure.^2)

                    Err_ebse["mean_g1"][i2, :], Err_ebse["var_g1"][i2, :], counters["ebse_g1"] =
                        calc_recursive(Err_ebse["mean_g1"][i2, :], Err_ebse["var_g1"][i2, :],
                                        counters["ebse_g1"], ebse_measure[:, idx_ebse].^2)

                end
            end
        end
    end
end
#y_max1 = maximum([maximum(Err_bpf[:, :, 1]), maximum(Err_apf[:, :, 1])])
#y_max2 = maximum([maximum(Err_bpf[:, :, 2]), maximum(Err_apf[:, :, 2])])

#y_max1_measure = maximum([maximum(Err_bpf_measure[:, :, 1]), maximum(Err_apf_measure[:, :, 1])])
#y_max2_measure = maximum([maximum(Err_bpf_measure[:, :, 2]), maximum(Err_apf_measure[:, :, 2])])

#x_axis = "eventkernel" # particles, eventkernel


"""
if x_axis == "particles"
    figure(1)
    clf()
    for i2 = 1:n
        subplot(2, n, i2)
        plot(N[:], Err_bpf[:, i2, 1], "o")
        plot(N[:], Err_apf[:, i2, 1], "o")
        ylim([0, y_max1])
        title(Δ[i2])
    end
    for i2 = 1:n
        subplot(2, n, n+i2)
        plot(N[:], Err_apf[:, i2, 2], "o")
        plot(N[:], Err_bpf[:, i2, 2], "o")
        ylim([0, y_max2])
        title(Δ[i2])
    end

    figure(2)
    clf()
    for i2 = 1:n
        subplot(2, n, i2)
        plot(N[:], Err_bpf_measure[:, i2, 1], "x")
        plot(N[:], Err_apf_measure[:, i2, 1], "x")
        ylim([0, y_max1_measure])
        title(Δ[i2])
    end
    for i2 = 1:n
        subplot(2, n, n+i2)
        plot(N[:], Err_bpf_measure[:, i2, 2], "x")
        plot(N[:], Err_apf_measure[:, i2, 2], "x")
        ylim([0, y_max2_measure])
        title(Δ[i2])
    end
elseif x_axis == "eventkernel"
    figure(1)
    clf()
    for i1 = 1:m
        subplot(2, m, i1)
        plot(Δ[:], Err_bpf[i1, :, 1], "o")
        plot(Δ[:], Err_apf[i1, :, 1], "o")
        ylim([0, y_max1])
        title(N[i1])
    end
    for i1 = 1:m
        subplot(2, m, m+i1)
        plot(Δ[:], Err_bpf[i1, :, 2], "o")
        plot(Δ[:], Err_apf[i1, :, 2], "o")
        ylim([0, y_max2])
        title(N[i1])
    end

    figure(2)
    clf()
    for i1 = 1:m
        subplot(2, m, i1)
        plot(Δ[:], Err_bpf_measure[i1, :, 1], "x")
        plot(Δ[:], Err_apf_measure[i1, :, 1], "x")
        ylim([0, y_max1_measure])
        title(N[i1])
    end
    for i1 = 1:m
        subplot(2, m, m+i1)
        plot(Δ[:], Err_bpf_measure[i1, :, 2], "x")
        plot(Δ[:], Err_apf_measure[i1, :, 2], "x")
        ylim([0, y_max2_measure])
        title(N[i1])
    end
else
    print("No such x_axis setting\n")
end
"""

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
s = [sqrt.(Err_bpf["var"][N_lags, :, 1]),
     sqrt.(Err_apf["var"][N_lags, :, 1]),
     sqrt.(Err_ebse["var"][:, 1]),
     sqrt.(Err_kalman["var"][1])]
plot(Δ[:], m[1], "C0x-")
plot(Δ[:], m[2], "C1o-")
plot(Δ[:], m[3], "C2^-")
plot(Δ[:], m[4]*ones(n), "r--")
fill_between(Δ[:], m[1] - s[1], m[1] + s[1], facecolor="C0", alpha=0.3)
fill_between(Δ[:], m[2] - s[2], m[2] + s[2], facecolor="C1", alpha=0.3)
fill_between(Δ[:], m[3] - s[3], m[3] + s[3], facecolor="C2", alpha=0.3)
fill_between(Δ[:], m[4] - s[4], m[4] + s[4], facecolor="r", alpha=0.3)
legend(["BPF", "APF", "EBSE", "Kalman"])
subplot(1, 2, 2)
title("State x_2")
m = [Err_bpf["mean"][N_lags, :, 2],
     Err_apf["mean"][N_lags, :, 2],
     Err_ebse["mean"][:, 2],
     Err_kalman["mean"][2]]
s = [sqrt.(Err_bpf["var"][N_lags, :, 2]),
     sqrt.(Err_apf["var"][N_lags, :, 2]),
     sqrt.(Err_ebse["var"][:, 2]),
     sqrt.(Err_kalman["var"][2])]
plot(Δ[:], m[1], "C0x-")
plot(Δ[:], m[2], "C1o-")
plot(Δ[:], m[3], "C2^-")
plot(Δ[:], m[4]*ones(n), "r--")
fill_between(Δ[:], m[1] - s[1], m[1] + s[1], facecolor="C0", alpha=0.3)
fill_between(Δ[:], m[2] - s[2], m[2] + s[2], facecolor="C1", alpha=0.3)
fill_between(Δ[:], m[3] - s[3], m[3] + s[3], facecolor="C2", alpha=0.3)
fill_between(Δ[:], m[4] - s[4], m[4] + s[4], facecolor="r", alpha=0.3)

figure(2)
clf()
subplot(1, 2, 1)
title("State x_1")
m = [Err_bpf["mean_g1"][N_lags, :, 1],
     Err_apf["mean_g1"][N_lags, :, 1],
     Err_ebse["mean_g1"][:, 1],
     Err_kalman["mean"][1]]
s = [sqrt.(Err_bpf["var_g1"][N_lags, :, 1]),
     sqrt.(Err_apf["var_g1"][N_lags, :, 1]),
     sqrt.(Err_ebse["var_g1"][:, 1]),
     sqrt.(Err_kalman["var"][1])]
plot(Δ[:], m[1], "C0x-")
plot(Δ[:], m[2], "C1o-")
plot(Δ[:], m[3], "C2^-")
plot(Δ[:], m[4]*ones(n), "r--")
fill_between(Δ[:], m[1] - s[1], m[1] + s[1], facecolor="C0", alpha=0.3)
fill_between(Δ[:], m[2] - s[2], m[2] + s[2], facecolor="C1", alpha=0.3)
fill_between(Δ[:], m[3] - s[3], m[3] + s[3], facecolor="C2", alpha=0.3)
fill_between(Δ[:], m[4] - s[4], m[4] + s[4], facecolor="r", alpha=0.3)
legend(["BPF", "APF", "EBSE", "Kalman"])
subplot(1, 2, 2)
title("State x_2")
m = [Err_bpf["mean_g1"][N_lags, :, 2],
     Err_apf["mean_g1"][N_lags, :, 2],
     Err_ebse["mean_g1"][:, 2],
     Err_kalman["mean"][2]]
s = [sqrt.(Err_bpf["var_g1"][N_lags, :, 2]),
     sqrt.(Err_apf["var_g1"][N_lags, :, 2]),
     sqrt.(Err_ebse["var_g1"][:, 2]),
     sqrt.(Err_kalman["var"][2])]
plot(Δ[:], m[1], "C0x-")
plot(Δ[:], m[2], "C1o-")
plot(Δ[:], m[3], "C2^-")
plot(Δ[:], m[4]*ones(n), "r--")
fill_between(Δ[:], m[1] - s[1], m[1] + s[1], facecolor="C0", alpha=0.3)
fill_between(Δ[:], m[2] - s[2], m[2] + s[2], facecolor="C1", alpha=0.3)
fill_between(Δ[:], m[3] - s[3], m[3] + s[3], facecolor="C2", alpha=0.3)
fill_between(Δ[:], m[4] - s[4], m[4] + s[4], facecolor="r", alpha=0.3)

figure(3)
clf()
subplot(1, 2, 1)
title("State x_1")
"""
m = [log.(Err_bpf["mean_g1"][:, Δ_lags, 1]),
     log.(Err_apf["mean_g1"][:, Δ_lags, 1]),
     log.(Err_ebse["mean_g1"][Δ_lags, 1]),
     log.(Err_kalman["mean"][1])]
s = [log.(sqrt.(Err_bpf["var_g1"][:, Δ_lags, 1])),
     log.(sqrt.(Err_apf["var_g1"][:, Δ_lags, 1])),
     log.(sqrt.(Err_ebse["var_g1"][Δ_lags, 1])),
     log.(sqrt.(Err_kalman["var"][1]))]
"""
m = [Err_bpf["mean_g1"][:, Δ_lags, 1],
     Err_apf["mean_g1"][:, Δ_lags, 1],
     Err_ebse["mean_g1"][Δ_lags, 1],
     Err_kalman["mean"][1]]
s = [sqrt.(Err_bpf["var_g1"][:, Δ_lags, 1]),
     sqrt.(Err_apf["var_g1"][:, Δ_lags, 1]),
     sqrt.(Err_ebse["var_g1"][Δ_lags, 1]),
     sqrt.(Err_kalman["var"][1])]
plot(N[:], m[1], "C0x-")
plot(N[:], m[2], "C1o-")
plot(N[:], m[3]*ones(length(N)), "C2^-")
plot(N[:], m[4]*ones(length(N)), "r--")
fill_between(N[:], m[1] - s[1], m[1] + s[1], facecolor="C0", alpha=0.3)
fill_between(N[:], m[2] - s[2], m[2] + s[2], facecolor="C1", alpha=0.3)
fill_between(N[:], m[3] - s[3], m[3] + s[3], facecolor="C2", alpha=0.3)
fill_between(N[:], m[4] - s[4], m[4] + s[4], facecolor="r", alpha=0.3)
legend(["BPF", "APF", "EBSE", "Kalman"])
subplot(1, 2, 2)
title("State x_2")
m = [log.(Err_bpf["mean_g1"][:, Δ_lags, 2]),
     log.(Err_apf["mean_g1"][:, Δ_lags, 2]),
     log.(Err_ebse["mean_g1"][Δ_lags, 2]),
     log.(Err_kalman["mean"][2])]
s = [log.(sqrt.(Err_bpf["var_g1"][:, Δ_lags, 2])),
     log.(sqrt.(Err_apf["var_g1"][:, Δ_lags, 2])),
     log.(sqrt.(Err_ebse["var_g1"][Δ_lags, 2])),
     log.(sqrt.(Err_kalman["var"][2]))]
plot(N[:], m[1], "C0x-")
plot(N[:], m[2], "C1o-")
plot(N[:], m[3]*ones(length(N)), "C2^-")
plot(N[:], m[4]*ones(length(N)), "r--")
fill_between(N[:], m[1] - s[1], m[1] + s[1], facecolor="C0", alpha=0.3)
fill_between(N[:], m[2] - s[2], m[2] + s[2], facecolor="C1", alpha=0.3)
fill_between(N[:], m[3] - s[3], m[3] + s[3], facecolor="C2", alpha=0.3)
fill_between(N[:], m[4] - s[4], m[4] + s[4], facecolor="r", alpha=0.3)
#legend(["BPF", "APF", "EBSE", "Kalman"])

#legend(["BPF", "SIS", "APF", "EBSE", "Kalman"])
"""
title(@sprintf("Delta: %d", Δ[Δ_lags]))
plot(N[:], Err_bpf[:, Δ_lags, 1], "x-")
plot(N[:], Err_apf[:, Δ_lags, 1], "x-")
plot(N[:], Err_ebse[Δ_lags, 1]*ones(m), "x-")
plot(N[:], Err_kalman[1]*ones(m), "r--")
legend(["BPF", "APF", "EBSE", "Kalman"])
subplot(4, 1, 2)
plot(N[:], Err_bpf[:, Δ_lags, 2], "x-")
plot(N[:], Err_apf[:, Δ_lags, 2], "x-")
plot(N[:], Err_ebse[Δ_lags, 2]*ones(m), "x-")
plot(N[:], Err_kalman[2]*ones(m), "r--")
legend(["BPF", "APF", "EBSE", "Kalman"])
figure(3)
clf()
subplot(2, 1, 1)
plot(1:L, Err_bpf_lags[N_lags, Δ_lags, :, 1])
#plot(1:L, Err_sis_lags[N_lags, Δ_lags, :, 1])
plot(1:L, Err_apf_lags[N_lags, Δ_lags, :, 1])
plot(1:L, Err_ebse_lags[Δ_lags, :, 1])
legend(["BPF", "APF", "EBSE"])
#legend(["BPF", "SIS", "APF", "EBSE"])
subplot(2, 1, 2)
plot(1:L, Err_bpf_lags[N_lags, Δ_lags, :, 2])
#plot(1:L, Err_sis_lags[N_lags, Δ_lags, :, 2])
plot(1:L, Err_apf_lags[N_lags, Δ_lags, :, 2])
plot(1:L, Err_ebse_lags[Δ_lags, :, 2])
legend(["BPF", "APF", "EBSE"])
"""

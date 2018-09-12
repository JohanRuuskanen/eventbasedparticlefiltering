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

if read_new
    Err_bpf = zeros(m, n, 2)
    Err_apf = zeros(m, n, 2)
    Err_ebse = zeros(n, 2)
    Err_kalman = zeros(2)
    
    Err_bpf_measure = zeros(m, n, 2)
    Err_apf_measure = zeros(m, n, 2)
    Err_ebse_measure = zeros(n, 2)

    Err_bpf_lags = zeros(m, n, L, 2)
    Err_apf_lags = zeros(m, n, L, 2)
    Err_ebse_lags = zeros(m, L, 2)


    function calc_lag_error(measure, counter, Γ)
        
        lag_error = zeros(length(counter), 2)
        idx_set = find(x -> x == 1, Γ)
        
        for l = 1:length(counter)
            for idx = 1:length(idx_set)
                if idx < length(idx_set)
                    if idx_set[idx] + l <= idx_set[idx+1]
                        lag_error[l, :] += (measure[:, idx_set[idx] + l - 1]).^2
                        counter[l] += 1
                    end
                else
                    if idx_set[idx] + l <= length(Γ)
                        lag_error[l, :] += (measure[:, idx_set[idx] + l - 2]).^2
                        counter[l] += 1
                    end
                end
            end
        end
        return lag_error, counter
    end


    for i1 = 1:m
        for i2 = 1:n
            print(string(i1)*" "*string(i2))

            E = load("/home/johanr/projects/EBPF/test_linear/data/test_linear_system/sim_" * string(i1) * "_" * string(i2) * ".jld", "results")

            if i1 == 1
                E_ref = load("/home/johanr/projects/EBPF/test_linear/data/test_linear_system_ref/sim_" * string(i2) * ".jld", "results")
            end

            counter_bpf = zeros(L)
            counter_apf = zeros(L)
            counter_ebse = zeros(L)
            for i3 = 1:k
                bpf_measure = E[i3]["err_ebpf"]
                apf_measure = E[i3]["err_eapf"]

                Γ_bpf = E[i3]["trigg_ebpf"]
                Γ_apf = E[i3]["trig_eapf"]

                idx_bpf = find(x -> x == 1, Γ_bpf)
                idx_apf = find(x -> x == 1, Γ_apf)

                Err_bpf[i1, i2, :] += mean(bpf_measure.^2, 2)
                Err_apf[i1, i2, :] += mean(apf_measure.^2, 2)

                Err_bpf_measure[i1, i2, :] += mean(bpf_measure[:, idx_bpf].^2, 2)
                Err_apf_measure[i1, i2, :] += mean(apf_measure[:, idx_apf].^2, 2)

                a, b =  calc_lag_error(bpf_measure, counter_bpf, Γ_bpf)
                Err_bpf_lags[i1, i2, :, :] += a
                counter_bpf = b
                
                a, b =  calc_lag_error(apf_measure, counter_apf, Γ_apf)
                Err_apf_lags[i1, i2, :, :] += a
                counter_apf = b

                if i1 == 1
                    if i2 == 1
                        Err_kalman += mean(E_ref[i3]["err_kal"].^2, 2)
                    end
                    
                    ebse_measure = E_ref[i3]["err_ebse"]
                    Γ_ebse = E_ref[i3]["trig_ebse"]

                    idx_ebse = find(x -> x == 1, Γ_ebse)

                    Err_ebse[i2, :] += mean(ebse_measure.^2, 2)
                    Err_ebse_measure[i2, :] += mean(ebse_measure[:, idx_ebse].^2, 2)
                    
                    a, b = calc_lag_error(ebse_measure, counter_ebse, Γ_ebse)
                    Err_ebse_lags[i2, :, :] += a
                    counter_ebse = b
                end
            end

            Err_bpf[i1, i2, :] /= k
            Err_apf[i1, i2, :] /= k

            Err_bpf_measure[i1, i2, :] /= k
            Err_apf_measure[i1, i2, :] /= k

            for l = 1:L
                Err_bpf_lags[i1, i2, l, :] /= counter_bpf[l]
                Err_apf_lags[i1, i2, l, :] /= counter_apf[l]
            end

            if i1 == 1
                if i2 == 1
                    Err_kalman /= k
                end
                Err_ebse[i2, :] /= k
                Err_ebse_measure[i2, :] /= k
                for l = 1:L
                    Err_ebse_lags[i2, l, :] /= counter_ebse[l]
                end
            end

        end
    end
end
y_max1 = maximum([maximum(Err_bpf[:, :, 1]), maximum(Err_apf[:, :, 1])])
y_max2 = maximum([maximum(Err_bpf[:, :, 2]), maximum(Err_apf[:, :, 2])])

y_max1_measure = maximum([maximum(Err_bpf_measure[:, :, 1]), maximum(Err_apf_measure[:, :, 1])])
y_max2_measure = maximum([maximum(Err_bpf_measure[:, :, 2]), maximum(Err_apf_measure[:, :, 2])])

x_axis = "eventkernel" # particles, eventkernel



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

N_lags = 12
Δ_lags = 6

figure(3)
clf()
subplot(2, 1, 1)
title(@sprintf("particles: %d", N[N_lags]))
plot(Δ[:], Err_bpf[N_lags, :, 1], "x-")
plot(Δ[:], Err_apf[N_lags, :, 1], "x-")
plot(Δ[:], Err_ebse[:, 1], "x-")
legend(["BPF", "APF", "EBSE"])
subplot(2, 1, 2)
plot(Δ[:], Err_bpf[N_lags, :, 2], "x-")
plot(Δ[:], Err_apf[N_lags, :, 2], "x-")
plot(Δ[:], Err_ebse[:, 2], "x-")
legend(["BPF", "APF", "EBSE"])

figure(4)
clf()
subplot(2, 1, 1)
title(@sprintf("particles: %d", N[N_lags]))
plot(Δ[:], Err_bpf_measure[N_lags, :, 1], "x-")
plot(Δ[:], Err_apf_measure[N_lags, :, 1], "x-")
plot(Δ[:], Err_ebse_measure[:, 1], "x-")
legend(["BPF", "APF", "EBSE"])
subplot(2, 1, 2)
plot(Δ[:], Err_bpf_measure[N_lags, :, 2], "x-")
plot(Δ[:], Err_apf_measure[N_lags, :, 2], "x-")
plot(Δ[:], Err_ebse_measure[:, 2], "x-")
legend(["BPF", "APF", "EBSE"])

figure(5)
clf()
subplot(2, 1, 1)
plot(1:L, Err_bpf_lags[N_lags, Δ_lags, :, 1])
plot(1:L, Err_apf_lags[N_lags, Δ_lags, :, 1])
plot(1:L, Err_ebse_lags[Δ_lags, :, 1])
legend(["BPF", "APF", "EBSE"])
subplot(2, 1, 2)
plot(1:L, Err_bpf_lags[N_lags, Δ_lags, :, 2])
plot(1:L, Err_apf_lags[N_lags, Δ_lags, :, 2])
plot(1:L, Err_ebse_lags[Δ_lags, :, 2])
legend(["BPF", "APF", "EBSE"])


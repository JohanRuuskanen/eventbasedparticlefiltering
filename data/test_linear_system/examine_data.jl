ENV["JULIA_PKGDIR"] = "/home/johanr/.julia"

using JLD
using PyPlot


N = [10 25 50 75 100 150 200 250 300 350 400 500]
Δ = [0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.8 4.0]

m = length(N)
n = length(Δ)
k = 1000

# MSE
Err_bpf = zeros(m, n, 2)
Err_apf = zeros(m, n, 2)
Err_bpf_measure = zeros(m, n, 2)
Err_apf_measure = zeros(m, n, 2)

for i1 = 1:m
    for i2 = 1:n
        print(string(i1)*" "*string(i2))

        E = load("/home/johanr/projects/EBPF/data/test_linear_system/sim_" * string(i1) * "_" * string(i2) * ".jld", "results")

        for i3 = 1:k
            Err_bpf[i1, i2, :] += mean(E[i3]["err_ebpf"].^2, 2)
            Err_apf[i1, i2, :] += mean(E[i3]["err_eapf"].^2, 2)


            bpf_measure = E[i3]["err_ebpf"][:, find(x-> x ==1, E[i3]["trigg_ebpf"])]
            apf_measure = E[i3]["err_eapf"][:, find(x-> x ==1, E[i3]["trig_eapf"])]
            Err_bpf_measure[i1, i2, :] += mean(bpf_measure.^2, 2)
            Err_apf_measure[i1, i2, :] += mean(apf_measure.^2, 2)
        end

        Err_bpf[i1, i2, :] /= k
        Err_apf[i1, i2, :] /= k

        Err_bpf_measure[i1, i2, :] /= k
        Err_apf_measure[i1, i2, :] /= k

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

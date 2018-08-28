ENV["JULIA_PKGDIR"] = "/home/johanr/.julia"

using JLD
using PyPlot


N = [10 50 100 150 200 250 300 350 400 450 500 600 700 800 900 1000 2000 3000 5000]
Δ = [0]

m = length(N)
n = length(Δ)
k = 1000

# MSE
Err_ebpf = zeros(m, n, 2)
Err_eapf = zeros(m, n, 2)
Err_bpf = zeros(m, n, 2)
Err_apf = zeros(m, n, 2)

for i1 = 1:m
    for i2 = 1:n
        E = load("/home/johanr/projects/EBPF/data/test_results_over_N/sim_" * string(i1) * "_" * string(i2) * ".jld", "results")

        for i3 = 1:k
            Err_ebpf[i1, i2, :] += mean(E[i3]["err_ebpf"].^2, 2)
            Err_eapf[i1, i2, :] += mean(E[i3]["err_eapf"].^2, 2)
            Err_bpf[i1, i2, :] += mean(E[i3]["err_bpf"].^2, 2)
            Err_apf[i1, i2, :] += mean(E[i3]["err_apf"].^2, 2)
        end

        Err_ebpf[i1, i2, :] /= k
        Err_eapf[i1, i2, :] /= k
        Err_bpf[i1, i2, :] /= k
        Err_apf[i1, i2, :] /= k

    end
end

y_max1 = maximum([maximum(Err_ebpf[:, :, 1]), maximum(Err_eapf[:, :, 1]),
        maximum(Err_bpf[:, :, 1]), maximum(Err_apf[:, :, 1])])
y_max2 = maximum([maximum(Err_ebpf[:, :, 2]), maximum(Err_eapf[:, :, 2]),
        maximum(Err_bpf[:, :, 2]), maximum(Err_apf[:, :, 2])])

x_axis = "particles" # particles, eventkernel

if x_axis == "particles"
    figure(1)
    for i2 = 1:n
        subplot(2, n, i2)
        plot(N[:], Err_ebpf[:, i2, 1], "bo", alpha=0.5)
        plot(N[:], Err_eapf[:, i2, 1], "ro", alpha=0.5)
        plot(N[:], Err_bpf[:, i2, 1], "bx", alpha=0.5)
        plot(N[:], Err_apf[:, i2, 1], "rx", alpha=0.5)
        ylim([0, y_max1])
        title(Δ[i2])
    end
    for i2 = 1:n
        subplot(2, n, n+i2)
        plot(N[:], Err_ebpf[:, i2, 2], "bo", alpha=0.5)
        plot(N[:], Err_eapf[:, i2, 2], "ro", alpha=0.5)
        plot(N[:], Err_bpf[:, i2, 2], "bx", alpha=0.5)
        plot(N[:], Err_apf[:, i2, 2], "rx", alpha=0.5)
        ylim([0, y_max2])
        title(Δ[i2])
    end
else
    print("No such x_axis setting\n")
end

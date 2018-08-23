using JLD
using PyPlot

E = load("/home/johanr/projects/EBPF/data/mc_data1.jld", "results")

m, n, k = size(E)

N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Δ = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]

# MSE for the EBPF and the EAPF
Err_bpf = zeros(m, n, 2)
Err_apf = zeros(m, n, 2)

Err_bpf_measure = zeros(m, n, 2)
Err_apf_measure = zeros(m, n, 2)



for i1 = 1:m
    for i2 = 1:n
        for i3 = 1:k
            Err_bpf[i1, i2, :] += mean(E[i1, i2, i3]["err_ebpf"].^2, 2) 
            Err_apf[i1, i2, :] += mean(E[i1, i2, i3]["err_eapf"].^2, 2) 
        
            
            bpf_measure = E[i1, i2, i3]["err_ebpf"][:, find(x-> x ==1, E[i1, i2, i3]["trig_ebpf"])]
            apf_measure = E[i1, i2, i3]["err_eapf"][:, find(x-> x ==1, E[i1, i2, i3]["trig_eapf"])]
            
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

figure(1)
for i2 = 1:n
    subplot(2, n, i2)
    plot(N, Err_bpf[:, i2, 1], "o")
    plot(N, Err_apf[:, i2, 1], "o")
    ylim([0, y_max1])
    title(Δ[i2])
end
for i2 = 1:n
    subplot(2, n, n+i2)
    plot(N, Err_bpf[:, i2, 2], "o")
    plot(N, Err_apf[:, i2, 2], "o")
    ylim([0, y_max2])
    title(Δ[i2])
end

figure(2)
for i2 = 1:n
    subplot(2, n, i2)
    plot(N, Err_bpf_measure[:, i2, 1], "x")
    plot(N, Err_apf_measure[:, i2, 1], "x")
    ylim([0, y_max1_measure])
    title(Δ[i2])
end
for i2 = 1:n
    subplot(2, n, n+i2)
    plot(N, Err_bpf_measure[:, i2, 2], "x")
    plot(N, Err_apf_measure[:, i2, 2], "x")
    ylim([0, y_max2_measure])
    title(Δ[i2])
end

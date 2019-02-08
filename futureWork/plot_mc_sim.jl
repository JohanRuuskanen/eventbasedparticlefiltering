using JLD
using PyPlot

data = load("/local/home/johanr/projects/EBPF/data_sim.jld")

err_nbpf = data["err_nbpf"]
err_zbpf = data["err_zbpf"]
err_nbpf_trigg = data["err_nbpf_trigg"]
err_zbpf_trigg = data["err_zbpf_trigg"]

s, m, n = size(err_nbpf)

En = zeros(m, n)
Ez = zeros(m, n)
En_t = zeros(m, n)
Ez_t = zeros(m, n)

for k = 1:m
    for i = 1:n

        En[k, i] = mean(err_nbpf[:, k, i])
        Ez[k, i] = mean(err_zbpf[:, k, i])

        En_t[k, i] = mean(err_nbpf_trigg[:, k, i])
        Ez_t[k, i] = mean(err_zbpf_trigg[:, k, i])

    end
end

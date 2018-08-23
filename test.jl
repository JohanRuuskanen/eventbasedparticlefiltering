using JLD
using PyPlot
using StatsBase
using Distributions


include("src/misc.jl")
include("src/event_kernels.jl")
include("src/particle_filters.jl")
include("src/particle_filters_eventbased.jl")

# Parameters
N = 100
T = 100
δ1 = 5
δ2 = 1

# Nonlinear and non-Gaussian system
f(x, t) = MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos(1.2*t), 10*eye(1))
h(x, t) = MvNormal(x.^2/20, 0.1*eye(1)) #MvNormal(atan.(x), 1*eye(1))
nd = [1, 1]

sys = sys_params(f, h, T, nd)
x, y = sim_sys(sys)

# For estimation
par = pf_params(N)

#X_bpf, W_bpf = bpf(y, sys, par)
#X_sir, W_sir = sir(y, sys, par)
#X_apf, W_apf = apf(y, sys, par)


xh_naive, yh_naive, Z_naive, Γ_naive = naive_filter(y, sys, δ1)
X_nbpf, W_nbpf, xh_nbpf, yh_nbpf, Z_nbpf, Γ_nbpf = ebpf_naive(y, sys, par, δ1)
X_zbpf, W_zbpf, xh_zbpf, yh_zbpf, Z_zbpf, Γ_zbpf = ebpf_usez(y, sys, par, δ1)


#xh_bpf = zeros(nd[1], T)
#xh_sir = zeros(nd[1], T)
#xh_apf = zeros(nd[1], T)
xh_nbpf2 = zeros(nd[1], T)
xh_zbpf2 = zeros(nd[1], T)
for k = 1:nd[1]
    #xh_bpf[k, :] = sum(diag(W_bpf'*X_bpf[:, k, :]), 2)
    #xh_sir[k, :] = sum(diag(W_sir'*X_sir[:, k, :]), 2)
    #xh_apf[k, :] = sum(diag(W_apf'*X_apf[:, k, :]), 2)
    xh_nbpf2[k, :] = sum(diag(W_nbpf'*X_nbpf[:, k, :]), 2)
    xh_zbpf2[k, :] = sum(diag(W_zbpf'*X_zbpf[:, k, :]), 2)
end


#err_bpf = zeros(nd[1], T)
#err_sir = zeros(nd[1], T)
#err_apf = zeros(nd[1], T)
err_naive = zeros(nd[1], T)
err_nbpf = zeros(nd[1], T)
err_zbpf = zeros(nd[1], T)
for k = 1:nd[1]
    #err_bpf[k, :] = x[k, :] - xh_bpf[k, :]
    #err_sir[k, :] = x[k, :] - xh_sir[k, :]
    #err_apf[k, :] = x[k, :] - xh_apf[k, :]
    err_naive[k, :] = x[k, :] - xh_naive[k, :]
    err_nbpf[k, :] = x[k, :] - xh_nbpf[k, :]
    err_zbpf[k, :] = x[k, :] - xh_zbpf[k, :]
end

println("")
println("RMSE naive: $(sqrt(mean(err_naive.^2)))")
#println("RMSE bpf: $(sqrt(mean(err_bpf.^2)))")
#println("RMSE sir: $(sqrt(mean(err_sir.^2)))")
#println("RMSE apf: $(sqrt(mean(err_apf.^2)))")
println("RMSE nbpf: $(sqrt(mean(err_nbpf.^2)))")
println("RMSE zbpf: $(sqrt(mean(err_zbpf.^2)))")
println("")


ytop = 1.2*maximum(y)
ybot = 1.2*minimum(y)
xtop = 1.2*maximum(x)
etop = 30#1.2*maximum(err_naive)

cmap = get_cmap("viridis")
levels = plt[:MaxNLocator](nbins=100)[:tick_values](0, 1)

#=
# The BPF
figure(1)
clf()
subplot(3, 2, 1)
plot(1:T, y')
ylim([ybot, ytop])
legend("y")
title("Measurements bpf")
subplot(3, 2, 3)
plot(1:T, x')
plot(1:T, xh_bpf', "r--")
ylim([-xtop, xtop])
legend(["x", "xh"])
title("State")
subplot(3, 2, 5)
plot(1:T, abs.(err_bpf'))
ylim([0, etop])
legend("err")
title("Error")
subplot(3, 2, 2)
range = -30:1:31
posterior_est = zeros(length(range)-1, T)
for k = 1:T
    posterior_est[:, k] = fit(Histogram, X_bpf[:, 1, k], weights(W_bpf[:, k]), range, closed=:left).weights
end
contourf(1:T, range[1:end-1], posterior_est, levels=levels, cmap=cmap)
# p1 particle histogram
# calculate unique particles
subplot(3, 2, 4)
bpf_unique = zeros(T)
for k = 1:T
    bpf_unique[k] = length(unique(X_bpf[:, 1, k])) / N
end
plot(1:T, bpf_unique)
title("Fraction of unqiue particles")
subplot(3,2,6)
plot(1:T, abs.(xh_bpf' - xh_bpf'))
ylim([0, etop])
legend(["xh", "xh_pf"])
title("Compare estimate with particle estimate")

# The sir_PF
figure(2)
clf()
subplot(3, 2, 1)
plot(1:T, y')
ylim([ybot, ytop])
legend("y")
title("Measurements sir")
subplot(3, 2, 3)
plot(1:T, x')
plot(1:T, xh_sir', "r--")
ylim([-xtop, xtop])
legend(["x", "xh"])
title("State")
subplot(3, 2, 5)
plot(1:T, abs.(err_sir'))
ylim([0, etop])
legend("err")
title("Error")
subplot(3, 2, 2)
range = -30:1:31
posterior_est = zeros(length(range)-1, T)
for k = 1:T
    posterior_est[:, k] = fit(Histogram, X_sir[:, 1, k], weights(W_sir[:, k]), range, closed=:left).weights
end
contourf(1:T, range[1:end-1], posterior_est, levels=levels, cmap=cmap)
# p1 particle histogram
# calculate unique particles
subplot(3, 2, 4)
sir_unique = zeros(T)
for k = 1:T
    sir_unique[k] = length(unique(X_sir[:, 1, k])) / N
end
plot(1:T, bpf_unique)
title("Fraction of unqiue particles")
subplot(3,2,6)
plot(1:T, abs.(xh_sir' - xh_sir'))
ylim([0, etop])
legend(["xh", "xh_pf"])
title("Compare estimate with particle estimate")
=#

# Naive filter
figure(2)
clf()
subplot(3, 1, 1)
plot(1:T, y')
plot(1:T, yh_naive', "r--")
plt[:step](1:T, Z_naive', where="post")
ylim([ybot, ytop])
legend(["y", "yh", "z"])
title("Measurements, naive")
subplot(3, 1, 2)
plot(1:T, x')
plot(1:T, xh_naive', "r--")
ylim([-xtop, xtop])
legend(["x", "xh"])
title("State")
subplot(3, 1, 3)
plot(1:T, abs.(err_naive'))
ylim([0, etop])
legend("err")
title("Error")

# Naive EBPF
figure(3)
clf()
subplot(3, 2, 1)
plot(1:T, y')
plot(1:T, yh_nbpf', "r--")
plt[:step](1:T, Z_nbpf', where="post")
ylim([ybot, ytop])
legend(["y", "yh", "z"])
title("Measurements, nbpf")
subplot(3, 2, 3)
plot(1:T, x')
plot(1:T, xh_nbpf', "r--")
ylim([-xtop, xtop])
legend(["x", "xh"])
title("State")
subplot(3, 2, 5)
plot(1:T, abs.(err_nbpf'))
ylim([0, etop])
legend("err")
title("Error")
subplot(3, 2, 2)
range = -30:1:31
posterior_est = zeros(length(range)-1, T)
for k = 1:T
    #posterior_est[:, k] = fit(Histogram, X_nbpf[:, 1, k], weights(W_nbpf[:, k]), range, closed=:left).weights
    x_tmp = X_nbpf[:, 1, k]
    posterior_est[:, k] = fit(Histogram, x_tmp, weights(ones(length(x_tmp), 1)), range, closed=:left).weights
end
logpost = log.(posterior_est)
logpost[find(x->x==-Inf,logpost)] = 0
contourf(1:T, range[1:end-1], logpost, levels=levels, cmap=cmap)
subplot(3, 2, 4)
nbpf_unique = zeros(T)
for k = 1:T
    nbpf_unique[k] = length(unique(X_nbpf[:, 1, k])) / N
end
plot(1:T, nbpf_unique)
title("Fraction of unqiue particles")
subplot(3,2,6)
plot(1:T, abs.(xh_nbpf' - xh_nbpf2'))
ylim([0, etop])
legend(["xh", "xh_pf"])
title("Compare estimate with particle estimate")

# EBPF that use Z
figure(4)
clf()
subplot(3, 2, 1)
plot(1:T, y')
plot(1:T, yh_zbpf', "r--")
plt[:step](1:T, Z_zbpf', where="post")
ylim([ybot, ytop])
legend(["y", "yh", "z"])
title("Measurements, zbpf")
subplot(3, 2, 3)
plot(1:T, x')
plot(1:T, xh_zbpf', "r--")
ylim([-xtop, xtop])
legend(["x", "xh"])
title("State")
subplot(3, 2, 5)
plot(1:T, abs.(err_zbpf'))
ylim([0, etop])
legend("err")
title("Error")
subplot(3, 2, 2)
range = -30:1:31
posterior_est = zeros(length(range)-1, T)
for k = 1:T
    #posterior_est[:, k] = fit(Histogram, X_nbpf[:, 1, k], weights(W_nbpf[:, k]), range, closed=:left).weights
    x_tmp = X_zbpf[:, 1, k]
    posterior_est[:, k] = fit(Histogram, x_tmp, weights(ones(length(x_tmp), 1)), range, closed=:left).weights
end
logpost = log.(posterior_est)
logpost[find(x->x==-Inf,logpost)] = 0
contourf(1:T, range[1:end-1], logpost, levels=levels, cmap=cmap)
subplot(3, 2, 4)
zbpf_unique = zeros(T)
for k = 1:T
    zbpf_unique[k] = length(unique(X_zbpf[:, 1, k])) / N
end
plot(1:T, zbpf_unique)
title("Fraction of unqiue particles")
subplot(3,2,6)
plot(1:T, abs.(xh_zbpf' - xh_zbpf2'))
ylim([0, etop])
legend(["xh", "xh_pf"])
title("Compare estimate with particle estimate")

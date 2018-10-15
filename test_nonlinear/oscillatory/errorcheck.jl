
# include("examine_system.jl")
#

idx1_s = sortperm(abs(err1_t), rev=true)
idx2_s = sortperm(abs(err2_t), rev=true)
k1 = idx1_s[6]
k2 = idx2_s[6]

figure(1)
clf()
subplot(2, 1, 1)
plt[:hist](X1[:, 1, idx1[k1]], bins=50, density=true, alpha=0.5)
plt[:hist](X1[:, 1, idx1[k1]], bins=50, density=true, weights=W1[:, idx1[k1]], alpha=0.5)
plot(x[idx1[k1]], 1, "r*") 

subplot(2, 1, 2)
plt[:hist](X2[:, 1, idx2[k2]], bins=50, density=true, alpha=0.5)
plt[:hist](X2[:, 1, idx2[k2]], bins=50, density=true, weights=W2[:, idx2[k2]], alpha=0.5)
plot(x[idx2[k2]], 1, "r*") 

#=
idx1_c = []
idx2_c = []
for i = 1:length(
figure(2)
clf()
subplot(2, 1, 1)
plt[:hist](X1[:, 1, idx1[k1]], bins=50, density=true, alpha=0.5)
plt[:hist](X1[:, 1, idx1[k1]], bins=50, density=true, weights=W1[:, idx1[k1]], alpha=0.5)
plot(x[idx1[k1]], 1, "r*") 

subplot(2, 1, 2)
plt[:hist](X2[:, 1, idx2[k2]], bins=50, density=true, alpha=0.5)
plt[:hist](X2[:, 1, idx2[k2]], bins=50, density=true, weights=W2[:, idx2[k2]], alpha=0.5)
plot(x[idx2[k2]], 1, "r*") 
=#

figure(3)
clf()
subplot(2, 1, 1)
plt[:hist](err1_t, density=true)
subplot(2, 1, 2)
plt[:hist](err2_t, bins=100, density=true)


println("X1 Error at $(k): $(err1_t[k1])")
println("X2 Error at $(k): $(err2_t[k2])")




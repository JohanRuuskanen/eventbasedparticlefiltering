using PyPlot
using Distributions

f(x, t) = MvNormal(x/2 + 25*x ./ (1 + x.^2) + 8*cos(1.2*t), 10*eye(1))
h(x, t) = MvNormal(atan.(x), 1*eye(1))


x = [5]
t = [1]

z = linspace(-20, 20, 1000)

fp = zeros(size(z))
hp = zeros(size(z))
for k = 1:length(z)
    fp[k] = pdf(f(x, t), [z[k]])
    hp[k] = pdf(h(x, t), [z[k]])
end

figure(1)
clf()
subplot(3, 1, 1)
plot(z, fp)
title("Propagation distribution")
subplot(3, 1, 2)
plot(z, hp)
title("Measurement distribution")
subplot(3, 1, 3)

title("Approximate ")

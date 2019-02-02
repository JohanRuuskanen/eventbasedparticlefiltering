
function EventKer_MBT(xh, z, y, k, δ, sys)

    # Propagate estimate
    xh = mean(sys.f(xh, k))
    yh = mean(sys.h(xh, k)) + var(sys.f(xh, k))/20
    γ = [0]

    # Trigger sampling on bad estimate
    if norm(yh - y) >= δ
        z = y
        γ = [1]
    end

    return xh, yh, z, γ
end

function EventKer_SOD(xh, z, y, k, δ, sys)

    γ = 0

    if norm(z - y) >= δ
        z = y
        γ = 1
    end

    return xh, z, z, γ

end

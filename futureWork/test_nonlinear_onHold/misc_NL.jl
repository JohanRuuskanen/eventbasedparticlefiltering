struct sys_params
    f::Function
    h::Function
    w
    v
    T
    nd
end

function sim_sys(sys)
    x = zeros(sys.nd[1], sys.T)
    y = zeros(sys.nd[2], sys.T)

    for k = 2:sys.T
        x[:, k] = sys.f(x[:, k-1], k) + rand(sys.w)
        y[:, k] = sys.h(x[:, k], k) + rand(sys.v)
    end

    return x, y
end

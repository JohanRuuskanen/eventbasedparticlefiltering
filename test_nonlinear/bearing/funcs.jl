struct pf_params
    N
    X0
end

struct sys_params
    f
    h
    w
    v
    T
    nd
end

function fix_sym(Σ; lim=1e-9)
    # Caclulating the inverse yields small numerical errors that makes the
    # matrices non-symmetric by a small margin.

    Σ_new = Array(Hermitian(Σ))

    if norm(Σ - Σ_new) > lim
        println("Warning: norm difference in symmetrization exceedes limit $(lim); $(norm(Σ - Σ_new))")
    end
    return Σ_new
end

function sim_sys(sys, x0)
    x = zeros(sys.nd[1], sys.T)
    y = zeros(sys.nd[2], sys.T)

    x[:, 1] = x0
    #y[:, 1] = sys.h(x0) + rand(sys.v)
    y[:, 1] = sys.h(x0, 1) + rand(sys.v)

    #BP(x) = exp(7/x) + exp(7/(1000 - x)) - 2

    for k = 2:sys.T

        x[:, k] = sys.f(x[:, k-1], k) + rand(sys.w)

        #=
        x[2, k] += sign(500 - x[1, k])*BP(x[1, k])
        x[4, k] += sign(500 - x[3, k])*BP(x[3, k])

        if x[1, k] < 0
            x[1, k] = 1
            x[2, k] = 0
        elseif x[1, k] > 1000
            x[1, k] = 999
            x[2, k] = 0
        end

        if x[3, k] < 0
            x[3, k] = 1
            x[4, k] = 0
        elseif x[4, k] > 1000
            x[3, k] = 999
            x[4, k] = 0
        end
        =#

        y[:, k] = sys.h(x[:, k], k) + rand(sys.v)
    end

    return x, y
end

function generate_H1H2(x, h, k)
    #h1(x) = sqrt(x[1]^2 + x[3]^2)
    #h2(x) = atan(x[1]/x[3])

    h1(x) = h(x, k)[1]
    #h2(x) = h(x, k)[2]

    H(x) = [h1(x)]#[h1(x), h2(x)]
    H1 = ForwardDiff.jacobian(H, x)

    H2 = ForwardDiff.hessian(h1, x)
            #[  diag(ForwardDiff.hessian(h1, x))'];
            #diag(ForwardDiff.hessian(h2, x))']

    return H1, H2
end

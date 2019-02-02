
struct sys_params
    f::Function
    h::Function
    w
    v
    T
    nd
end

struct lin_sys_params
    A::Array{Float64, 2}
    C::Array{Float64, 2}
    Q::Array{Float64, 2}
    R::Array{Float64, 2}
    T::Int64
end


struct pf_params
    N::Number
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

function sim_lin_sys(sys)
    nx = size(sys.A, 1)
    ny = size(sys.C, 1)

    x = zeros(nx, sys.T)
    y = zeros(ny, sys.T)

    for k = 2:sys.T
        x[:, k] = sys.A * x[:, k-1] + rand(MvNormal(zeros(nx), sys.Q))
        y[:, k] = sys.C * x[:, k] + rand(MvNormal(zeros(ny), sys.R))
    end

    return x, y
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

struct sys_params
    A::Array{Float64, 2}
    C::Array{Float64, 2}
    Q::Array{Float64, 2}
    R::Array{Float64, 2}
    T::Int64
end

struct output
    X::Array{Float64, 3}
    W::Array{Float64, 2}
    S::Array{Float64, 2}
    Z::Array{Float64, 2}
    Γ::Array{Float64, 1}
    res::Array{Float64, 1}
    fail::Array{Float64, 1}
end

struct pf_params
    N::Int64
    eventKernel::String
    Nlim::Float64
end

function sim_sys(sys::sys_params)
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

function fix_sym(Σ::Array{Float64,2}; lim=1e-9, warning=true)
    # Caclulating the inverse yields small numerical errors that makes the
    # matrices non-symmetric by a small margin.

    Σ_new = Array(Hermitian(Σ))

    if norm(Σ - Σ_new) > lim && warning
        println("Warning: norm difference in symmetrization exceedes limit $(lim); $(norm(Σ - Σ_new))")
    end
    return Σ_new
end

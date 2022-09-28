# Kernel structs
@with_kw struct kernel_SOD{T <: Number}
    δ::Array{T, 1}
end

@with_kw struct kernel_IBT{T <: Number}
    δ::Array{T, 1}
end

# Particle filter type structs
struct pftype_bootstrap
end

@with_kw mutable struct pftype_auxiliary
    qv::Function
    q::Function
    D::Integer
end


# Numerical inegration scheme structs
struct likelihood_analytic
end

@with_kw mutable struct likelihood_MC
    M::Integer
end

@with_kw mutable struct likelihood_cubature
    reltol::Float64 = 1e-9
end


@with_kw mutable struct system
    px::Function
    py::Function
    nx::Integer
    ny::Integer
    t_end::Integer
end

@with_kw mutable struct ebpf_options
    sys::system
    N::Integer
    kernel::Union{kernel_SOD, kernel_IBT}
    pftype::Union{pftype_bootstrap, pftype_auxiliary}
    likelihood::Union{likelihood_analytic, likelihood_MC, likelihood_cubature} = likelihood_analytic()
    triggerat::String = "events"
    debug_save::Bool = false
    abort_at_trig::Bool = false
    print_progress::Bool = true
    predictive_computation::Bool = false
    extra_params::Dict = Dict()
end

@with_kw mutable struct particle_data{T <: AbstractFloat}
    X::Array{T, 3}
    Xr::Array{T, 3}
    Xp::Array{T, 3}
    W::Array{T, 2}
    V::Array{T, 2}
    H::Array{T, 3}
    Hh::Array{Distribution, 2}
    qv_list::Array{T, 3}
    q_list::Array{Distribution, 2}
    S::Array{Integer, 2}
    Γ::Array{Bool, 1}
    fail::Array{Bool, 1}
    p_trig::Array{T, 1}
    extra::Dict{Any, Any} = Dict()
end

struct err_result{T <: AbstractFloat}
    err_all::Array{T, 1}
    err_trig::Array{T, 1}
    err_noTrig::Array{T, 1}
    ce_all::T
    ce_trig::T
    ce_noTrig::T
    ce_errs::Int64
    nbr_trig::Int64
    nbr_frac::T
    effectiveness::Array{T, 1}
    filtered_nans::Int64
end

function generate_pfd(opt::ebpf_options; T=Float64)

    @assert T <: AbstractFloat

    if opt.debug_save
        t_extra = opt.sys.t_end
    else
        t_extra = 1
    end

    if typeof(opt.pftype) <: pftype_auxiliary
        D = opt.pftype.D
    else
        D = 1
    end

    pfd = particle_data(    X = zeros(T, opt.N, opt.sys.nx, opt.sys.t_end),
                            Xr = zeros(T, opt.N, opt.sys.nx, t_extra),
                            Xp = zeros(T, opt.N, opt.sys.nx, t_extra),
                            W = zeros(T, opt.N, opt.sys.t_end) ./ opt.N,
                            V = zeros(T, opt.N, t_extra),
                            H = zeros(T, opt.sys.ny, 2, opt.sys.t_end),
                            Hh = Array{Distribution, 2}(undef, D, t_extra),
                            qv_list = ones(T, D, opt.N, t_extra),
                            q_list = Array{Distribution, 2}(undef, opt.N, t_extra),
                            S = zeros(Int64, opt.N, opt.sys.t_end),
                            Γ = zeros(Bool, opt.sys.t_end),
                            fail = zeros(Bool, opt.sys.t_end),
                            p_trig = zeros(T, opt.sys.t_end)
                        )
    return pfd
end

function estimate_py_pred!(pfd::particle_data, k::Integer,
        opt::ebpf_options) where T <: AbstractFloat
    kh = opt.debug_save ? k : 1

    idx = resampling_systematic(pfd.W[:, k-1])

    pfd.S[:, k] .= idx

    Y = Array{Distribution, 1}(undef, opt.N)
    for i = 1:opt.N
        pfd.Xp[i, :, kh] = rand(opt.sys.px(pfd.X[idx[i], :, k-1], k-1))
        Y[i] = opt.sys.py(pfd.Xp[i, :, kh], k)
    end
    return Y
end

function sim_sys(sys::system; T=Float64, x0=Array{Float64,1}())

    @assert T <: AbstractFloat

    x = zeros(T, sys.nx, sys.t_end)
    y = zeros(T, sys.ny, sys.t_end)

    if !isempty(x0)
        x[:, 1] .= x0
    end
    y[:, 1] .=  rand(sys.py(x[:, 1], 1))

    # TODO: Here it should be k-1 on both in px
    for k = 2:sys.t_end
        x[:, k] .= rand(sys.px(x[:, k-1], k-1))
        y[:, k] .= rand(sys.py(x[:, k], k))
    end

    return x, y
end

function fix_sym(Σ::Array{T,2}; lim=1e-9, warning=true) where T <: AbstractFloat
    # Caclulating the inverse sometimes yields small numerical errors that makes
    # the matrices non-symmetric by a small margin.

    Σ_new = Array(Hermitian(Σ))

    if norm(Σ - Σ_new) > lim && warning
        println("Warning: norm difference in symmetrization exceedes limit $(lim); $(norm(Σ - Σ_new))")
    end
    return Σ_new
end

function run_example(file)
    current_path = @__DIR__
    example_path = current_path[1:end-9] * "examples/"
    include(example_path*file)
end

function compute_err_metrics(pfd::particle_data, x::AbstractArray{T,2}) where T <: AbstractFloat

    t_end = size(pfd.X, 3)
    nx = size(pfd.X, 2)
    xh = zeros(T, nx, t_end)
    for i = 1:nx
        xh[i, :] = sum(pfd.W .* pfd.X[:, i, :], dims=1)
    end
    contains_nan = isnan.(sum(xh, dims=1))[:]
    filtered_nans = sum(contains_nan)

    if filtered_nans > 0
        println("Warning: $(filtered_nans) NaN values detected, removing")
    end

    x_clean = x[:, .!contains_nan]
    xh_clean = xh[:, .!contains_nan]
    Γ_clean = pfd.Γ[.!contains_nan]

    X_clean = pfd.X[:, :, .!contains_nan]
    W_clean = pfd.W[:, .!contains_nan]

    # Metrics
    idx0 = findall(x -> x == 0, Γ_clean)
    idx1 = findall(x -> x == 1, Γ_clean)

    # Standard
    err_all = mean((x_clean - xh_clean).^2, dims=2)[:]
    err_1 =  mean((x_clean[:, idx1] - xh_clean[:, idx1]).^2, dims=2)[:]
    err_0 =  mean((x_clean[:, idx0] - xh_clean[:, idx0]).^2, dims=2)[:]
    nbr_1 = sum(Int64, pfd.Γ)
    frac = T(nbr_1 / length(pfd.Γ))
    err_trig = err_all*nbr_1

    # Cross-entropy
    if nx <= 2
      lh = zeros(T, size(x_clean, 2))
      if nx == 1
        for k = 1:size(x_clean, 2)
          p = kde(X_clean[:, 1, k], weights=Weights(W_clean[:, k]))
          lh[k] = pdf(p, x_clean[k])
        end
      elseif nx == 2
        for k = 1:size(x_clean, 2)
          p = kde(X_clean[:, :, k], weights=Weights(W_clean[:, k]))
          lh[k] = pdf(p, x_clean[1, k], x_clean[2, k])
        end
      end

      idx_lh_err = findall(x -> x <= 0, lh)

      deleteat!(lh, idx_lh_err)
      deleteat!(Γ_clean, idx_lh_err)

      idx0 = findall(x -> x == 0, Γ_clean)
      idx1 = findall(x -> x == 1, Γ_clean)

      ce_all = - mean(log.(lh))
      ce_1 = - mean(log.(lh[idx1]))
      ce_0 = - mean(log.(lh[idx0]))
      ce_errs = length(idx_lh_err)
    else
      println("Cross-entropy not implemented for nx > 2, as KernelDensity is not")
      ce_all = T(0.0)
      ce_1 = T(0.0)
      ce_0 = T(0.0)
      ce_errs = 0
    end

    return err_result(err_all, err_1, err_0, ce_all, ce_1, ce_0, ce_errs,
                      nbr_1, frac, err_trig, filtered_nans)

end


function generate_example_system(T; type="linear_ny1")

    if type == "linear_ny1"
        nx = 2
        ny = 1

        A = [0.2 0.8; 0 0.95]
        C = [1.0 0]

        Q = 1*Matrix{Float64}(I, nx, nx)
        R = 0.1*Matrix{Float64}(I, ny, ny)

        px = (x, k) -> MvNormal(A*x, Q)
        py = (x, k) -> MvNormal(C*x, R)

        p0 = MvNormal(ones(nx), 1.0)

    elseif type == "linear_ny2"
        nx = 3
        ny = 2

        A = [   0.9 0.2 -0.2;
                0 0.95 0.0;
                0.0 0.0 0.95]
        C = [   0.5 1.0 0.0;
                0.5 0.0 1.0]

        Q = 1*Matrix{Float64}(I, nx, nx)
        R = 0.1*Matrix{Float64}(I, ny, ny)

        px = (x, k) -> MvNormal(A*x, Q)
        py = (x, k) -> MvNormal(C*x, R)

        p0 = MvNormal(ones(nx), 1.0)

    elseif type == "nonlinear_ny2"
        nx = 4
        ny = 2

        A = [   1.0 0.2 0.0 0.0;
                0.0 0.0 0.0 0.0;
                0.0 0.0 1.0 0.2;
                0.0 0.0 0.0 0.0]

        C = zeros(ny, nx)

        Q = [   2.0 0.0 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 2.0 0.0;
                0.0 0.0 0.0 1.0]
        R = [   0.1 0.0;
                0.0 0.01]

        px = (x, k) -> MvNormal(A*x + [0, 10*sin(2*pi/100*k), 0, 10*cos(2*pi/100*k)], Q)
        py = (x, k) -> MvNormal([sqrt(x[1]^2 + x[3]^2), atan(x[1]/x[3])], R)

        p0 = MvNormal([100.0, 0.0, 100.0, 0.0], 1.0)

    elseif type == "nonlinear_SOD_pathological"

        nx = 1
        ny = 1

        A = Matrix{Float64}(undef, nx, nx)
        A[1,1] = 0.99
        C = zeros(ny, nx)

        Q = 0.2*Matrix{Float64}(I, nx, nx)
        R = 0.1*Matrix{Float64}(I, ny, ny)

        px = (x, k) -> MvNormal(A*x, Q)
        py = (x, k) -> MvNormal(5*cos.(2*pi/10 * k .+ x), R)

        p0 = MvNormal(nx, 1.0)

    elseif type == "nonlinear_classic"

        nx = 1
        ny = 1

        A = zeros(nx, nx)
        C = zeros(nx, nx)

        Q = 1*Matrix{Float64}(I, nx, nx)
        R = 0.1*Matrix{Float64}(I, ny, ny)

        px = (x, k) -> MvNormal(x/2 + 25*x./(1 .+ x.^2) .+ 8*cos(1.2*k), Q)
        py = (x, k) -> MvNormal(x.^2/20, R)

        p0 = MvNormal(nx, 1.0)

    else
        error("No such system")
    end

    sys = system(px, py, nx, ny, T)

    return sys, A, C, Q, R, p0
end

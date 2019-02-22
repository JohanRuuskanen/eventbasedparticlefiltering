
function calculate_weights!(W::AbstractArray{T,1}, X::AbstractArray{T,2},
    z::AbstractArray{T,1}, Wr::AbstractArray{T,1}, Xr::AbstractArray{T,2},
    yh, q_list, q_aux_list, r, γ, δ, sys, par) where T <: Real

    M = size(yh,1)

    if γ == 1
        for i = 1:par.N
            if r == 1
                W[i] = log(Wr[i]) + log(pdf(MvNormal(sys.C*X[i, :], sys.R), z)) +
                            log(pdf(MvNormal(sys.A*Xr[i, :], sys.Q), X[i, :])) -
                            log(pdf(q_list[i], X[i, :])) -
                            log(pdf(q_aux_list[i], z))
            else
                W[i] = log(Wr[i]) + log(pdf(MvNormal(sys.C*X[i, :], sys.R), z)) +
                            log(pdf(MvNormal(sys.A*Xr[i, :], sys.Q), X[i, :])) -
                            log(pdf(q_list[i], X[i, :]))
            end
        end
    else
        for i = 1:par.N
            # Likelihood state constrained
            D = Normal((sys.C*X[i, :])[1], sys.R[1])
            yp = sys.C * X[i, :]
            if norm(z - yp) < δ
                py = pdf(D, yp[1])
            else
                py = 0
            end

            # Propagation
            px = pdf(MvNormal(sys.A*Xr[i, :], sys.Q), X[i, :])

            # proposal distribution
            q = pdf(q_list[i], X[i, :])

            # Predictive likelihood
            pL = 0
            for j = 1:M
                pL += pdf(q_aux_list[i], yh[j, :])
            end
            pL /= M

            if r == 1
                W[i] = log(Wr[i]) + log(py) + log(px) - log(pL) - log(q)
            else
                W[i] = log(Wr[i]) + log(py) + log(px) - log(q)
            end

        end
    end
end

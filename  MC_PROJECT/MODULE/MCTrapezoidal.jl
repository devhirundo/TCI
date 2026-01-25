module MCTrapezoidal

using ..MeshBase
using Statistics
using Random

export integrate

function get_weight(indices, grids)
    weight = 1.0
    for (i, idx) in enumerate(indices)
        if idx == 1 || idx == grids[i]
            weight *= 0.5
        end
    end
    return weight
end

function integrate(mf::MeshFunction{N}, samples::Int) where N
    epochs = 10
    estimates = zeros(Float64, epochs)

    n_total = prod(mf.grids)
    dv = get_volume_element(mf)
    discrete_vol = n_total * dv

    for e in 1:epochs
        local_weighted_sum = 0.0
        for _ in 1:samples
            idxs = ntuple(d -> rand(1:mf.grids[d]), N)

            w = get_weight(idxs, mf.grids)

            local_weighted_sum += w * mf[idxs...]
        end
        estimates[e] = (local_weighted_sum / samples) * discrete_vol
    end
    
    return mean(estimates), std(estimates) / sqrt(samples)
end

end
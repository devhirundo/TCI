module TrapezoidalIntegral
using ..MeshBase
export integrate

function integrate(mf::MeshFunction{N}) where N
    total_weighted_sum = 0.0
    for ci in CartesianIndices(mf.grids)
        idx_tuple = Tuple(ci)
        weight = 1.0
        for i in 1:N
            if idx_tuple[i] == 1 || idx_tuple[i] == mf.grids[i]
                weight *= 0.5
            end
        end
        total_weighted_sum += weight * mf[ci]
    end
    return total_weighted_sum * get_volume_element(mf)
end
end
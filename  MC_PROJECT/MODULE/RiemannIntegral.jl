module RiemannIntegral
using ..MeshBase
export integrate

function integrate(mf::MeshFunction{N}) where N
    total_sum = 0.0
    for ci in CartesianIndices(mf.grids)
        total_sum += mf[ci]
    end
    return total_sum * get_volume_element(mf)
end
end
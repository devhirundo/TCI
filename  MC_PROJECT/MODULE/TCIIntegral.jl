#TCI using trapezoidal Rule
module TCIIntegral

using ..MeshBase
using TensorCrossInterpolation

export integrate

function integrate(mf::MeshFunction{N}; tol=1e-12) where N
    eval_count = Ref(0)
    
    function wrapper_func(indices)
        eval_count[] += 1
        return mf[indices...]
    end

    tci, ranks, errors = crossinterpolate2(Float64, wrapper_func, mf.grids; tolerance=tol)
    
    trap_weights = ntuple(d -> begin
        w = ones(mf.grids[d])
        w[1] = 0.5
        w[end] = 0.5
        w
    end, N)

    tt = TensorTrain(tci)
    weighted_sum = sum(tt, collect(trap_weights))
    
    dv = get_volume_element(mf)
    return weighted_sum * dv, eval_count[]
end

end
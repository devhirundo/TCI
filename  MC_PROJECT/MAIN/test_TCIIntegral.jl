include("../MODULE/MeshBase.jl")
include("../MODULE/TCIIntegral.jl")

using .MeshBase
using .TCIIntegral
using TensorCrossInterpolation
using LinearAlgebra

function f(x1, x2, x3, x4, x5)
    return exp(x1 * x2 + x3 + x4 + x5)
end

N = 30 
domains = [(0.0, 1.0) for _ in 1:5]
grids = [N for _ in 1:5]

mf = MeshFunction(f, domains, grids)

function f_for_tci(idx)
    return mf[idx...]
end

localdims = [N, N, N, N, N]

tci, ranks, errors = crossinterpolate2(
    Float64,
    f_for_tci,
    localdims;
    tolerance=1e-8,
    maxiter=20
)
println("Bond Dimensions (Ranks): ", rank(tci))
println("Number of TT bond: ", length(ranks))

tensor_sum = sum(tci)
dv = get_volume_element(mf)
riemann_integral = tensor_sum * dv

println("Number of Grid (N^5): ", BigInt(N)^5)
println("sum of all elements in tensor: ", tensor_sum)
println("dv: ", dv)
println("Result of approximation for Riemann Sum : ", riemann_integral)

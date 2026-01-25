include("../MODULE/MeshBase.jl")
include("../MODULE/RiemannIntegral.jl")

using .MeshBase
using TensorCrossInterpolation
using LinearAlgebra
using Statistics
using Printf

function f(x1, x2, x3, x4, x5)
    return exp(x1 * x2 + x3 + x4 + x5)
end

N = 30
domains = [(0.0, 1.0) for _ in 1:5]
grids = [N for _ in 1:5]
num_mc_samples = 50000

mf = MeshFunction(f, domains, grids)

sitedims = [N^2, N, N, N] 

function f_grouped_tci(idx)
    i_grouped = idx[1]
    i3, i4, i5 = idx[2], idx[3], idx[4]
    
    x1_idx = (i_grouped - 1) % N + 1
    x2_idx = (i_grouped - 1) ÷ N + 1
    
    # MeshBase getindex
    return mf[x1_idx, x2_idx, i3, i4, i5]
end

tci, ranks, errors = crossinterpolate2(
    Float64, f_grouped_tci, sitedims; 
    tolerance=1e-8
)

M3 = dropdims(sum(tci.sitetensors[2], dims=2), dims=2) # rank1 x rank2
M4 = dropdims(sum(tci.sitetensors[3], dims=2), dims=2) # rank2 x rank3
M5 = dropdims(sum(tci.sitetensors[4], dims=2), dims=2) # rank3 x 1

fixed_tail = M3 * M4 * M5 #rank1 x 1

mc_values = Float64[]

for _ in 1:num_mc_samples
    idx_bundled = rand(1:(N^2))
    
    core1_slice = tci.sitetensors[1][1, idx_bundled, :] 

    val = dot(core1_slice, fixed_tail)
    push!(mc_values, val)
end

dv = get_volume_element(mf)
mctci_integral = mean(mc_values) * (N^2) * dv

@printf("MCTCI Result : %.12f\n", mctci_integral)
println("Num of MC sample         : ", num_mc_samples)
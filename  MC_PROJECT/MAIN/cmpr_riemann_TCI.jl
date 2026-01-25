include("../MODULE/MeshBase.jl")
include("../MODULE/TCIIntegral.jl")
include("../MODULE/RiemannIntegral.jl")

using .MeshBase
using .TCIIntegral
using .RiemannIntegral
using TensorCrossInterpolation
using LinearAlgebra
using Printf

function f(x1, x2, x3, x4, x5)
    return exp(x1 * x2 + x3 + x4 + x5)
end

N = 30
domains = [(0.0, 1.0) for _ in 1:5]
grids = [N for _ in 1:5]

mf = MeshFunction(f, domains, grids)
dv = get_volume_element(mf)

tci_start_time = time()

f_tci(idx) = mf[idx...]

tci, ranks, errors = crossinterpolate2(
    Float64, f_tci, grids; 
    tolerance=1e-8, maxiter=20
)

tci_total_sum = sum(tci)
tci_riemann_val = tci_total_sum * dv

tci_end_time = time()
tci_duration = tci_end_time - tci_start_time
simple_start_time = time()

simple_riemann_val = RiemannIntegral.integrate(mf)

simple_end_time = time()
simple_duration = simple_end_time - simple_start_time

@printf("TCI based sum     : %.12f (time: %.4fs)\n", tci_riemann_val, tci_duration)
@printf("Excat sum    : %.12f (time: %.4fs)\n", simple_riemann_val, simple_duration)
println("-"^50)

abs_error = abs(tci_riemann_val - simple_riemann_val)
rel_error = abs_error / simple_riemann_val

@printf("Difference: %.2e\n", abs_error)
@printf("Rel. Error: %.2e\n", rel_error)
println("TCI Ranks      : ", rank(tci))
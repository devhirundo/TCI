include("../MODULE/MeshBase.jl")
include("../MODULE/TrapezoidalIntegral.jl")
include("../MODULE/MCTrapezoidal.jl")

using .MeshBase
import .TrapezoidalIntegral
import .MCTrapezoidal
using Plots
using Printf

const L = 1.0
const DIM = 6
const GRID_SIZE = 20
function f(x1, x2, x3, x4, x5)
    return exp(x1 * x2 + x3 + x4 + x5)
end

domains = [(0.0, L) for _ in 1:DIM]
grids = [GRID_SIZE for _ in 1:DIM]
mf = MeshFunction(f, domains, grids)

@time ref_val = TrapezoidalIntegral.integrate(mf)
println("ref value: $ref_val\n")

sample_counts = [10^3, 10^4, 10^5, 10^6]
mc_errors = Float64[]

for s in sample_counts
    mc_val, _ = MCTrapezoidal.integrate(mf, s)
    
    rel_err = abs(mc_val - ref_val) / ref_val * 100
    push!(mc_errors, rel_err)
    
    @printf("Samples: 10^%d | Relative Error to Trapezoid: %.6f %%\n", log10(s), rel_err)
end

p = plot(sample_counts, mc_errors,
    xaxis = :log10, yaxis = :log10,
    marker = :circle, lw = 2,
    label = "MC (Trapezoid Weighted)",
    xlabel = "Number of MC Samples",
    ylabel = "Relative Error to Full Sum (%)",
    title = "MC Convergence to Full Trapezoidal Sum (Grid 20^5)",
    grid = :both)

display(p)
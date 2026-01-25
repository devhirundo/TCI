include("../MODULE/MeshBase.jl")
include("../MODULE/AnalyticalIntegral.jl")
include("../MODULE/RiemannIntegral.jl")
include("../MODULE/TrapezoidalIntegral.jl")

using .MeshBase
import .AnalyticalIntegral
import .RiemannIntegral
import .TrapezoidalIntegral
using Plots
using Printf

const L = 1.0
const DIM = 5
function f(x1, x2, x3, x4, x5)
    return exp(x1 * x2 + x3 + x4 + x5)
end

grid_sizes = [5, 10, 15, 20, 25, 30]
exact_val = AnalyticalIntegral.get_exact_val(L)

riemann_errors = Float64[]
trapezoid_errors = Float64[]

for g in grid_sizes
    domains = [(0.0, L) for _ in 1:DIM]
    grids = [g for _ in 1:DIM]
    mf = MeshFunction(f, domains, grids)
    
    r_err = abs(RiemannIntegral.integrate(mf) - exact_val) / exact_val * 100
    t_err = abs(TrapezoidalIntegral.integrate(mf) - exact_val) / exact_val * 100
    
    push!(riemann_errors, r_err)
    push!(trapezoid_errors, t_err)
    @printf("Grid %d : Riemann %.2f%%, Trapezoid %.4f%%\n", g, r_err, t_err)
end

p = plot(grid_sizes, [riemann_errors, trapezoid_errors],
    label = ["Riemann Sum" "Trapezoidal Rule"],
    marker = [:circle :square],
    xaxis = :log10, yaxis = :log10,
    xlabel = "Grid Size (N per dimension)",
    ylabel = "Relative Error (%)",
    title = "Convergence Comparison (5D Integral)",
    lw = 2,
    grid = :both)

display(p)
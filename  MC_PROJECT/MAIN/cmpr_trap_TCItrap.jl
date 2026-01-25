include("../MODULE/MeshBase.jl")
include("../MODULE/TrapezoidalIntegral.jl")
include("../MODULE/TCIIntegral.jl")

using .MeshBase
import .TrapezoidalIntegral
import .TCIIntegral
using Plots
using Printf

const L = 1.0
const DIM = 5
const GRID_N = 20
function f(x1, x2, x3, x4, x5)
    return exp(x1 * x2 + x3 + x4 + x5)
end
mf = MeshFunction(f, [(0.0, L) for _ in 1:DIM], [GRID_N for _ in 1:DIM])

ref_val = TrapezoidalIntegral.integrate(mf)

tolerances = 10.0 .^ (-1:-1:-14)
errors = Float64[]
costs = Int[]

for tol in tolerances
    val, cost = TCIIntegral.integrate(mf, tol=tol)
    rel_err = abs(val - ref_val) / ref_val
    
    push!(errors, rel_err)
    push!(costs, cost)
    @printf("Tol: 1e-%-2d | Rel.Err: %.6e | Calls: %d\n", -log10(tol), rel_err, cost)
end

p = plot(tolerances, errors, xaxis=:log10, yaxis=:log10, xflip=true,
    marker=:circle, label="TCI vs Full Sum", title="TCI Convergence to Trapezoidal Sum")
display(p)
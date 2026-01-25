include("../MODULE/MeshBase.jl")
include("../MODULE/AnalyticalIntegral.jl")
include("../MODULE/RiemannIntegral.jl")
include("../MODULE/TrapezoidalIntegral.jl")

using .MeshBase
import .AnalyticalIntegral
import .RiemannIntegral
import .TrapezoidalIntegral
using Printf

const L = 1.0
const DIM = 5
function f(x1, x2, x3, x4, x5)
    return exp(x1 * x2 + x3 + x4 + x5)
end

exact_val = AnalyticalIntegral.get_exact_val(L)

println("Dimension: $DIM, Domain: 0~$L")
println("Exact: $exact_val")
@printf("%-10s | %-18s | %-18s\n", "Grid", "Riemann RelErr(%)", "Trapezoid RelErr(%)")

for g in [5, 10, 15, 20, 25, 30]
    domains = [(0.0, L) for _ in 1:DIM]
    grids = [g for _ in 1:DIM]
    mf = MeshFunction(f, domains, grids)
    
    r_val = RiemannIntegral.integrate(mf)
    t_val = TrapezoidalIntegral.integrate(mf)
    
    r_err = abs(r_val - exact_val) / exact_val * 100
    t_err = abs(t_val - exact_val) / exact_val * 100
    
    @printf("%-10d | %-18.6e | %-18.6e\n", g, r_err, t_err)
end
println("="^60)
# TCI_validation_long_range.jl
include("../MODULE/TCI.jl") 
using .TCI 
using LinearAlgebra

const L = 1

meshes = [60 for _ in 1:5] 
domains = [(0.0, L) for _ in 1:5]

function f(xs...)
    v = collect(xs)

    return exp(v[1]*v[2] + v[3] + v[4] + v[5])
end


function get_EXACT_VAL(L)
    sumVal = 0.0
    k = 0
    while true
        #Use big or bigInt for large number
        #Do not write as big(factorial(k)) -> due to the type casting, factorial(k) [Int] already diverge, it returns 0 (overflow)
        term = (L^(2*k + 2)) / (factorial(big(k)) * (k + 1)^2) 
        if term < 1e-12 #if term goes to 0
             break 
        end
        sumVal += Float64(term)
        k += 1
    end
    singleExp = (exp(L) - 1.0)^3
    return sumVal * singleExp
end

const EXACT_VAL = get_EXACT_VAL(L)
println("Exact Value", EXACT_VAL)

mf = TCI.MeshFunction(f, domains, meshes)

tt = TCI.run_tci(mf, tolerance=1e-12) 

integVal = TCI.integrate(mf, tt)
errorVal = abs(integVal - EXACT_VAL) / EXACT_VAL * 100

println("TCI Estimate : ", integVal)
println("Exact Value   : ", EXACT_VAL)
println("Relative Error(%) : ", errorVal, "%")


ranks = Int[]
push!(ranks, 1) #first core is always 1
for core in tt
    push!(ranks, size(core, 3))
end
println("Rank : ", join(ranks, " -> "))
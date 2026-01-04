include("../MODULE/SimplePlot.jl") 
include("../MODULE/TCI.jl") 
using .SimplePlot
using .TCI
using Plots
using Random
using Statistics
using Printf

const L = 1.0
const DIM = 5
const VOLUME = L^DIM 

const MESH_SIZE = 60
domains = [(0.0, L) for _ in 1:DIM]
meshes = [MESH_SIZE for _ in 1:DIM]


function f(xs...)
    v = collect(xs)
    return exp(v[1]*v[2] + v[3] + v[4] + v[5])
end

mf = TCI.MeshFunction(f, domains, meshes)

function get_exact_val(L)
    sumVal = 0.0
    k = 0
    while true
        #Use big or bigInt for large number
        term = (L^(2*k + 2)) / (factorial(big(k)) * (k + 1)^2) 
        if term < 1e-12 
             break 
        end
        sumVal += Float64(term)
        k += 1
    end
    singleExp = (exp(L) - 1.0)^3
    return sumVal * singleExp
end

const EXACT_VAL = get_exact_val(L)
println("Exact Value: ", EXACT_VAL)


stepLst = [i*500 for i in 1:100]
const EPOCHS = 100

x_data = Float64[] 
y_data = Float64[] 


@printf("%-5s | %-12s | %-12s | %-12s | %-10s\n", "Samples(N)", "Mean Estimate", "Rel Err(%)", "StdDev", "In 3-Sigma?")


for step in stepLst

    epochResults = Float64[]
    
    for _ in 1:EPOCHS
        localSum = 0.0

        coords = zeros(Float64, DIM)
        
        for _ in 1:step
            for d in 1:DIM
                randIdx = rand(1:mf.mesh_size[d])
                
                (minV, maxV) = mf.domain[d]
                points = mf.mesh_size[d]
                stepSize = (maxV - minV) / (points - 1)
                
                coords[d] = minV + (randIdx - 1) * stepSize
            end
            

            localSum += f(coords...)
        end
        
        est = VOLUME * (localSum / step)
        push!(epochResults, est)
    end

    meanEst = mean(epochResults)
    stdEpoch = std(epochResults)
    
    # In 3 sigma?
    lowerBound = meanEst - 3.0 * stdEpoch
    upperBound = meanEst + 3.0 * stdEpoch
    
    passOrFail = (lowerBound <= EXACT_VAL <= upperBound) ? "Pass" : "Fail"
    
    # Relative Error
    absDiff = abs(meanEst - EXACT_VAL)
    relError = (absDiff / EXACT_VAL) * 100.0

    @printf("%-5d | %-.8f | %-.4f %% | %-.4e | %s\n", step, meanEst, relError, stdEpoch, passOrFail)
        

    push!(x_data, step)
    push!(y_data, relError)
end


g_data = SimplePlot.GraphData(
    "Discrete MC Error (Epoch : $EPOCHS)", 
    "Number of Samples (N)",            
    "Relative Error (%)",               
    x_data,                             
    y_data                              
)

plt = SimplePlot.draw_graph(g_data)

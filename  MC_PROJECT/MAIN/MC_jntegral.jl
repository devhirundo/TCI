include("../MODULE/SimplePlot.jl") 
using .SimplePlot
using Plots
using Random
using Statistics
using Printf

const L = 1
const DIM = 5
const VOLUME = L^DIM 

function f(v::Vector{Float64})
    return exp(v[1]*v[2] + v[3] + v[4] + v[5])
end


function get_exact_val(L)
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

const EXACT_VAL = get_exact_val(L)
println("Exact Value", EXACT_VAL)


stepLst = [i*500 for i in 1:100]
const EPOCHS = 100

x_data = Float64[] 
y_data = Float64[] 


#Data align using @printf
# % : start formatting | - : start alignment from left | N : make N space | s : string type 
@printf("%-5s | %-12s | %-12s | %-12s | %-10s\n", "Samples(N)", "Mean Estimate", "Rel Err(%)", "StdDev", "In 3-Sigma?")

for step in stepLst

    epochResults = Float64[]
    
    for _ in 1:EPOCHS
        localSum = 0.0
        v = zeros(Float64, DIM)
        
        for _ in 1:step
            rand!(v) #0~1
            v .*= L  #0~L
            localSum += f(v)
        end
        
        est = VOLUME * (localSum / step)
        push!(epochResults, est)
    end
    
    #using Statistics pkg
    meanEst = mean(epochResults)
    stdEpoch = std(epochResults)
    
    # In 3 sigma?
    lowerBound = meanEst - 3.0 * stdEpoch
    upperBound = meanEst + 3.0 * stdEpoch
    
    passOrFail = (lowerBound <= EXACT_VAL <= upperBound) ? "Pass" : "Fail"
    
    #calculate relative error
    absDiff = abs(meanEst - EXACT_VAL)
    relError = (absDiff / EXACT_VAL) * 100.0
    
    #print out
    @printf("%-5d | %-.8f | %-.4f %% | %-.4e | %s\n", step, meanEst, relError, stdEpoch, passOrFail)
        
    #save data
    push!(x_data, step)
    push!(y_data, relError)
end


g_data = SimplePlot.GraphData(
    "MC Error (Epoch : $EPOCHS)", 
    "Number of Samples (N)",            
    "Relative Error (%)",               
    x_data,                             
    y_data                              
)

plt = SimplePlot.draw_graph(g_data)


#plot!(plt, yscale=:log10, label="MC Error")
#display(plt)
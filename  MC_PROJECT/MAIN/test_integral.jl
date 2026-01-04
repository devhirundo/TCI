using .Integral
using .SimplePlot

f(x) = sin(2 * pi * x)
domain = (0.0, pi)
grid = 100
mf = Integral.MeshFunction(f, domain, grid)

extVal = Integral.sum_all_elements(mf)
extValAbs = Integral.get_Z(mf) 

stepLst = collect(100:100:10000)
errValLst = Float64[]
resultLst = Float64[]
pfLst = Bool[]
epoch = 10

for step in stepLst
    println("\n==== mc step : $step ====")
    acc = 0.0; accSq = 0.0
    for i in 1:epoch
        mcSum = extValAbs * Integral.sum_sign_with_mcmc(mf, step, 100)
        acc += mcSum
        accSq += mcSum^2
    end
    result = acc / epoch
    errVal = sqrt(max(0.0, accSq / epoch - (acc / epoch)^2) / epoch)
    
    push!(resultLst, result)
    push!(errValLst, errVal)

    #println("MC Result : $result")
    #println("errVal    : $errVal")
    
    if (extVal - errVal) < result && result < (extVal + errVal)
        #println("Success :)")
        push!(pfLst,true)
    else
        #println("Fail :(")
        push!(pfLst,false)
    end
end

errValgraph = GraphData("MCMC Result", "mcmcStep", "errVal", log10.(stepLst), log10.(errValLst))
mcmcValgraph = GraphData("MCMC Result", "mcmcStep" , "Value", stepLst, resultLst)
pfgraph = GraphData("MCMC Result", "mcmcStep" , "pass - fail", stepLst, pfLst)


errValgraph = GraphData("MCMC Result", "mcmcStep", "errVal", log10.(stepLst), log10.(1 ./sqrt.(stepLst)))
errValgraph = GraphData("MCMC Result", "mcmcStep", "errVal", (stepLst), (errValLst))
draw_graph(errValgraph)
#draw_graph(pfgraph)

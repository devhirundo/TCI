using TensorCrossInterpolation
import TensorCrossInterpolation as TCI

function index_to_x(bitlist)
    return sum((bitlist .- 1) .* (0.5 .^ (1:length(bitlist))))
end

function x_to_index(x, R)
    x = clamp(x, 0.0, 0.99999)
    bitlist = ones(Int, R)
    for i in 1:R
        if x >= 0.5^i
            bitlist[i] = 2
            x -= 0.5^i
        end
    end
    return bitlist
end

function run_diffusion_simulation()
    R = 10                  
    localdims = fill(2, R)  
    dx = 1.0 / (2^R)        
    dt = 0.000004             
    alpha = 0.1             
    time_steps = 1000   

    function initial_condition(bitlist)
        x = index_to_x(bitlist)
        return exp(-100.0 * (x - 0.5)^2) 
    end

    firstpivots = [ones(Int, R)]
    tci_initial, _, _ = TCI.crossinterpolate2(
        Float64, initial_condition, localdims, firstpivots;
        tolerance=1e-6, maxbonddim=20
    )

    current_u_TT = TCI.TensorTrain(tci_initial)
    println("Initial Rank", maximum(TCI.linkdims(current_u_TT))) 

    for step in 1:time_steps

        function next_step_eval(bitlist)
            x = index_to_x(bitlist)
            
            idx_c = bitlist
            idx_l = x_to_index(x - dx, R)
            idx_r = x_to_index(x + dx, R)
            
            u_c = TCI.evaluate(current_u_TT, idx_c)
            u_l = TCI.evaluate(current_u_TT, idx_l)
            u_r = TCI.evaluate(current_u_TT, idx_r)
            
            d2u_dx2 = (u_r - 2u_c + u_l) / (dx^2)
            
            return u_c + dt * alpha * d2u_dx2
        end
        
        tci_next, _, _ = TCI.crossinterpolate2(
            Float64, next_step_eval, localdims, firstpivots;
            tolerance=1e-6, maxbonddim=30  
        )
        
        current_u_TT = TCI.TensorTrain(tci_next)
        
        if step % 100 == 0
            current_rank = maximum(TCI.linkdims(current_u_TT))
            center_val = TCI.evaluate(current_u_TT, x_to_index(0.5, R)) 
            println("Step $step | center_val: $(round(center_val, digits=4)) | max rank : $current_rank")
        end
    end
    
    return current_u_TT
end

final_state = run_diffusion_simulation()
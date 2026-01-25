module AnalyticalIntegral
export get_exact_val

function get_exact_val(L::Float64)
    sum_val = 0.0
    k = 0
    while true
        # L^(2k+2) / (k! * (k+1)^2)
        term = (L^(2*k + 2)) / (factorial(big(k)) * (k + 1)^2)
        if term < 1e-16 break end
        sum_val += Float64(term)
        k += 1
    end
    
    return sum_val * (exp(L) - 1.0)^3
end
end
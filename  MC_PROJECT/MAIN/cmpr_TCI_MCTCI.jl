include("../MODULE/MeshBase.jl")
include("../MODULE/RiemannIntegral.jl")

using .MeshBase
using TensorCrossInterpolation
using LinearAlgebra
using Statistics
using Plots
using Printf

const TRUE_VALUE = 8.005904388433
N = 30
domains = [(0.0, 1.0) for _ in 1:5]
grids = [N for _ in 1:5]
sample_counts = [100, 500, 1000, 5000, 10000, 50000]
num_trials = 10

function f(x1, x2, x3, x4, x5)
    return exp(x1 * x2 + x3 + x4 + x5)
end

mf = MeshFunction(f, domains, grids)
dv = get_volume_element(mf)

sitedims = [N^2, N, N, N]
function f_grouped(idx)
    i_grouped = idx[1]
    x1_idx = (i_grouped - 1) % N + 1
    x2_idx = (i_grouped - 1) ÷ N + 1
    return mf[x1_idx, x2_idx, idx[2], idx[3], idx[4]]
end

tci, _, _ = crossinterpolate2(Float64, f_grouped, sitedims; tolerance=1e-8)

M3 = dropdims(sum(tci.sitetensors[2], dims=2), dims=2)
M4 = dropdims(sum(tci.sitetensors[3], dims=2), dims=2)
M5 = dropdims(sum(tci.sitetensors[4], dims=2), dims=2)
fixed_tail = M3 * M4 * M5 # rank_1 x 1

final_means = Float64[]
standard_deviations = Float64[]

for M in sample_counts
    trial_results = Float64[]
    
    for t in 1:num_trials
        current_sum = 0.0

        samples = Float64[]
        
        for _ in 1:M
            idx_b = rand(1:(N^2))
            val = dot(tci.sitetensors[1][1, idx_b, :], fixed_tail)
            current_sum += val
            push!(samples, val)
        end
        
        integral_est = (current_sum / M) * (N^2) * dv
        push!(trial_results, integral_est)
    end
    
    m_val = mean(trial_results)
    std_val = std(trial_results) 
    se_val = std_val / sqrt(M) 
    
    push!(final_means, m_val)
    push!(standard_deviations, se_val)
    
    @printf("M: %6d | Mean: %.10f | SE: %.2e | Diff: %.2e\n", 
            M, m_val, se_val, abs(m_val - TRUE_VALUE))
end

p = plot(sample_counts, final_means, 
    yerror=standard_deviations, # stddev errbar
    xaxis=:log, 
    marker=:circle, 
    label="MC-TCI",
    title="MC-TCI",
    xlabel="Number of Sampling", 
    ylabel="MC-TCI Value",
    grid=true,
    lw=2)

hline!([TRUE_VALUE], line=(:dash, :red), label="True Value")

display(p)
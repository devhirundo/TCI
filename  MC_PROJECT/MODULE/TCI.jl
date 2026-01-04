# MODULE/TCI.jl

module TCI

    using TensorCrossInterpolation
    
    struct MeshFunction
        f::Function                    
        domain::Vector{Tuple{Float64, Float64}} 
        mesh_size::Vector{Int}            
    end

    #generator
    function MeshFunction(f::Function, doms::Vector{<:Tuple}, meshes::Vector{Int})
        return MeshFunction(f, [Float64.(d) for d in doms], meshes)
    end


    function run_tci(mf::MeshFunction; tolerance=1e-5)

        #index mapping
        function wrapper_func(indices)
            x_coords = Float64[]
            for (dim, idx) in enumerate(indices)
                (minVal, maxVal) = mf.domain[dim]
                points = mf.mesh_size[dim]
                step = (maxVal - minVal) / (points - 1)
                val = minVal + (idx - 1) * step
                push!(x_coords, val)
            end
            return mf.f(x_coords...) #slurping
        end

        tci, ranks, errors = TensorCrossInterpolation.crossinterpolate2(Float64,wrapper_func,mf.mesh_size;tolerance=tolerance)
        
        return tci
    end
    
    function integrate(mf::MeshFunction, tt)

        total_sum = sum(tt)
        
        dV = 1.0
        
        for i in 1:length(mf.domain)
            (minVal, maxVal) = mf.domain[i]
            points = mf.mesh_size[i]

            dV *= (maxVal - minVal) / points
        end
        
        return total_sum * dV
    end
end
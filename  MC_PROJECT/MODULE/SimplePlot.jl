module SimplePlot
    using Plots 
    
    export GraphData, HeatmapData, draw_graph, draw_heatmap

    struct GraphData
        title::String
        xlabel::String
        ylabel::String
        x::Vector{Float64}
        y::Vector{Float64}
    end

    # X, Y Both
    function GraphData(title::String, xlabel::String, ylabel::String, x::Vector, y::Vector)
        if length(x) != length(y)
            error("length(x) != length(y)")
        end
        return GraphData(title, xlabel, ylabel, Float64.(x), Float64.(y))
    end

    # Y (No X)
    function GraphData(title::String, xlabel::String, ylabel::String, y::Vector)
        N = length(y)
        generated_x = collect(1.0:N)
        
        return GraphData(title, xlabel, ylabel, generated_x, y)
    end


    function draw_graph(g::GraphData)
        plot(g.x, g.y, 
             title  = g.title,
             xlabel = g.xlabel,
             ylabel = g.ylabel,
             marker = :circle,   
             lw     = 2,         
             legend = false,     
             grid   = true       
        )
    end

    struct HeatmapData
        title::String
        xlabel::String
        ylabel::String
        x::Vector{Float64}
        y::Vector{Float64}
        z::Matrix{Float64}
    end

    function HeatmapData(title::String, xlabel::String, ylabel::String, x::Vector, y::Vector, f::Function)
        z = [f(xi, yi) for yi in y, xi in x]
        return HeatmapData(title, xlabel, ylabel, Float64.(x), Float64.(y), Float64.(z))
    end

    function draw_heatmap(h::HeatmapData)
        heatmap(h.x, h.y, h.z,
            title  = h.title,
            xlabel = h.xlabel,
            ylabel = h.ylabel,
            color  = :viridis,
            aspect_ratio = :equal
        )
    end

end
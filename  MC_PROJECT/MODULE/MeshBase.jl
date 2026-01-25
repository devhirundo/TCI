module MeshBase
export MeshFunction, get_volume_element

struct MeshFunction{N, F}
    f::F
    domains::NTuple{N, Tuple{Float64, Float64}}
    grids::NTuple{N, Int}
    steps::NTuple{N, Float64}
end

function MeshFunction(f::F, domains::Vector{Tuple{Float64, Float64}}, grids::Vector{Int}) where {F}
    N = length(domains)

    steps = ntuple(i -> (domains[i][2] - domains[i][1]) / (grids[i] - 1), N)

    return MeshFunction{N, F}(
        f, 
        ntuple(i -> domains[i], N), 
        ntuple(i -> grids[i], N), 
        steps
    )
end

function Base.getindex(mf::MeshFunction{N}, indices::Vararg{Int, N}) where N
    coords = ntuple(i -> mf.domains[i][1] + (indices[i] - 1) * mf.steps[i], N)
    return mf.f(coords...)
end

function Base.getindex(mf::MeshFunction{N}, ci::CartesianIndex{N}) where N
    return mf[Tuple(ci)...]
end

function get_volume_element(mf::MeshFunction{N}) where N
    return prod(mf.steps)
end

end
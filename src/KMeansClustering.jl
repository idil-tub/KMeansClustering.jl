module KMeansClustering

import Distributions.Distribution
import Distributions.Uniform
import Distributions.rand
import Random.AbstractRNG
import LinearAlgebra.norm
import Statistics.mean

const NonInteger = Core.Real

abstract type ClusterInit{V<:Union{<:NonInteger, AbstractArray{<:NonInteger}}} end

"""
    initialize(c::ClusterInit{V}, samples::AbstractVector{V}, k::Int64)::Vector{V}

Initializes `k` cluster centers from `samples` using the cluster initialization method `c`.

"""
function (c::ClusterInit{V})(samples::AbstractVector{V}, k::Int64)::Vector{V} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    error("Method initialize not implemented for $(typeof(c))")
end

"""
    initialize(c::UniformRandomInit{V}, samples::AbstractVector{V}, k::Int64)::Vector{V}

Initializes `k` cluster centers from `samples` using a uniform random distribution.
"""
struct UniformRandomInit{V<:Union{AbstractArray{<:NonInteger}, <:NonInteger}} <: ClusterInit{V} end

function (c::UniformRandomInit{V})(samples::AbstractVector{V}, k::Int64)::Vector{V} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    if samples isa AbstractVector{<:NonInteger}
        x = map(el -> [el], samples)
    else
        x = samples
    end
    
    dims = size(x[1])
    min_bounds = fill(typemax(T), dims...)
    max_bounds = fill(typemin(T), dims...)
    
    for sample in x
        min_bounds = min.(min_bounds, sample)
        max_bounds = max.(max_bounds, sample)
    end

    function generateSample(min::T, max::T)
        dist = Uniform{T}(min, max)
        return rand(dist)
    end

    ret = [collect(map(generateSample, min_bounds, max_bounds)) for _ in 1:k]
    
    if samples isa AbstractVector{<:NonInteger}
        ret = map(el -> el[1], x)
    end
    
    return ret
end

@enum KMeansAlgorithm begin
    Lloyd
end

"""
    KMeans(x::AbstractVector{V}, k::Int64; init::ClusterInit{V}=UniformRandomInit{V}(), max_iter=300, tol=0.0001, algorithm::KMeansAlgorithm=Lloyd)::Dict{V, Vector{V}}

Perform K-means clustering on the data `x` with `k` clusters.

Arguments:
- `x`: Input data as an abstract vector of type `V`.
- `k`: Number of clusters.
- `init`: Cluster initialization method. Default is `UniformRandomInit`.
- `max_iter`: Maximum number of iterations. Default is 300.
- `tol`: Tolerance for convergence. Default is 0.0001.
- `algorithm`: K-means algorithm to use. Default is `Lloyd`.

Returns a dictionary mapping each cluster center to its assigned samples.
"""
function KMeans(x::AbstractVector{V}, k::Int64; init::ClusterInit{V}=UniformRandomInit{V}(), max_iter=300, tol=0.0001, algorithm::KMeansAlgorithm=Lloyd)::Dict{V, Vector{V}} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    if length(x) == 0
        return Dict([])
    end
    if k <= 0
        throw(ArgumentError("k has to be > 0"))
    end
    centers = init(x, k)
    iter = 0
    err = typemax(T)

    clusters = []
    while iter < max_iter && err > tol
        clusters = buildClusters(x, centers)
        for i in 1:length(clusters)
            if length(clusters[i]) == 0
                clusters[i] = [rand(x)]
            end
        end
        new_centers = calculateCenter.(clusters)
        err = norm(centers .- new_centers)
        centers = new_centers
        iter += 1
    end
    return Dict(zip(centers, clusters))
end

"""
    buildClusters(xs::AbstractVector{V}, init::AbstractVector{V})::Vector{Vector{V}}

Assigns each sample in `xs` to the nearest cluster center in `init`.

Returns a vector of clusters, where each cluster is a vector of samples.
"""
function buildClusters(xs::AbstractVector{V}, init::AbstractVector{V})::Vector{Vector{V}} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    num_clusters = length(init)
    clusters = [Vector{V}() for _ in 1:num_clusters]
    for x in xs
        min_dst = typemax(T)
        min_index = 0
        for i in 1:num_clusters
            d = norm(x - init[i])
            if d < min_dst
                min_dst = d
                min_index = i
            end
        end
        push!(clusters[min_index], x)
    end
    return clusters
end

"""
    calculateCenter(xs::AbstractVector{V})::V

Calculates the center of the cluster `xs`.

Returns the calculated center.
"""
function calculateCenter(xs::AbstractVector{V})::V where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    return mean(xs)
end

# Exported from the module
export KMeans, ClusterInit, UniformRandomInit, buildClusters, calculateCenter

end  # module

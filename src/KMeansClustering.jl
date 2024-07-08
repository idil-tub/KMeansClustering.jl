module KMeansClustering

import Distributions.Distribution
import Distributions.Uniform
import Distributions.Categorical
import Distributions.rand
import Random.AbstractRNG
import LinearAlgebra.norm as la_norm
import Statistics.mean

const NonInteger = Core.Real


abstract type Norm{V<:Union{<:NonInteger, AbstractArray{<:NonInteger}}} end

"""
    (c::Norm{V})(x::V)::T where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Calculates the norm of `x`. Overwrite in your own subtypes of `Norm`.
"""
function (c::Norm{V})(x::V)::T where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    error("Method initialize not implemented for $(typeof(c))")
end

struct EuclideanNorm{V<:Union{<:NonInteger, AbstractArray{<:NonInteger}}} <: Norm{V} end
"""
    (c::EuclideanNorm{V})(x::V)::T where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Calculates the euclidean norm of `x`. 
"""
function (c::EuclideanNorm{V})(x::V)::T where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    return la_norm(x)
end

abstract type ClusterInit{V<:Union{<:NonInteger, AbstractArray{<:NonInteger}}} end

"""
    (c::ClusterInit{V})(samples::AbstractVector{V}, k::Int64, norm::Norm{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Initializes `k` cluster centers from `samples` using the cluster initialization method `c`. Overwrite in your subtypes of ClusterInit

"""
function (c::ClusterInit{V})(samples::AbstractVector{V}, k::Int64, norm::Norm{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    error("Method initialize not implemented for $(typeof(c))")
end

"""
    (c::UniformRandomInit{V})(samples::AbstractVector{V}, k::Int64, norm::Norm{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Initializes `k` cluster centers from `samples` using a uniform random distribution, over the bounding hyperrectangle of the samples
"""
struct UniformRandomInit{V<:Union{AbstractArray{<:NonInteger}, <:NonInteger}} <: ClusterInit{V} end

function (c::UniformRandomInit{V})(samples::AbstractVector{V}, k::Int64, norm::Norm{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
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
        ret = map(el -> el[1], ret)
    end
    return ret
end


struct KMeansPPInit{V<:Union{AbstractArray{<:NonInteger}, <:NonInteger}} <: ClusterInit{V} end

"""
    (c::KMeansPPInit{V})(samples::AbstractVector{V}, k::Int64, norm::Norm{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Perform K-means++ initialization to select initial cluster centers.

"""
function (c::KMeansPPInit{V})(samples::AbstractVector{V}, k::Int64, norm::Norm{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    centers = []
    n = length(samples)
    probabilities = fill(1/n, n)
    while length(centers) < k
        push!(centers, samples[rand(Categorical(probabilities))])
        total_distance = 0.0
        for (i, sample) in enumerate(samples)
            min_dist_sq = Inf
            for center in centers
                dist_sq = norm(sample-center)^2
                min_dist_sq = min(min_dist_sq, dist_sq)
            end
            probabilities[i] = min_dist_sq
            total_distance += min_dist_sq
        end
        probabilities /= total_distance
    end
    return centers
end


abstract type CentroidCalculator{V<:Union{<:NonInteger, AbstractArray{<:NonInteger}}} end
"""
    (c::CentroidCalculator{V})(samples::AbstractVector{V}, norm::Norm{V})::V where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Calculates the center from the `samples` using `norm`. Overwrite in your subtypes of CentroidCalculator
"""
function (c::CentroidCalculator{V})(samples::AbstractVector{V}, norm::Norm{V})::V where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    error("Method initialize not implemented for $(typeof(c))")
end

struct EuclideanMeanCentroid{V<:Union{<:NonInteger, AbstractArray{<:NonInteger}}} <: CentroidCalculator{V} end

"""
    (c::EuclideanMeanCentroid{V})(samples::AbstractVector{V}, norm::Norm{V})::V where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Calculates the center from the `samples` using the standard euclidean mean.
"""
function (c::EuclideanMeanCentroid{V})(samples::AbstractVector{V}, norm::Norm{V})::V where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    return mean(samples)
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
- `centroid`: Used to calculate center of each cluster. Default `EuclideanMeanCentroid`
- `norm`: Used to assign clusters to samples. Default `EuclideanNorm`

Returns a dictionary mapping each cluster center to its assigned samples.
"""
function KMeans(x::AbstractVector{V}, k::Int64; init::ClusterInit{V}=UniformRandomInit{V}(), max_iter=300, tol=0.0001, algorithm::KMeansAlgorithm=Lloyd, centroid::CentroidCalculator{V}=EuclideanMeanCentroid{V}(), norm::Norm{V}=EuclideanNorm{V}())::Dict{V, Vector{V}} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    if length(x) == 0
        return Dict([])
    end
    if k <= 0
        throw(ArgumentError("k has to be > 0"))
    end
    centers = init(x, k, norm)
    iter = 0
    err = typemax(T)

    clusters = []
    while iter < max_iter && err > tol
        clusters = buildClusters(x, centers, norm)
        for i in 1:length(clusters)
            if length(clusters[i]) == 0
                clusters[i] = [rand(x)]
            end
        end
        new_centers = [centroid(cluster, norm) for cluster in clusters]
        err = sum(norm.(centers .- new_centers))
        centers = new_centers
        iter += 1
    end
    return Dict(zip(centers, clusters))
end

"""
    buildClusters(xs::AbstractVector{V}, init::AbstractVector{V}, norm::Norm{V})::Vector{Vector{V}} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Assigns each sample in `xs` to the nearest cluster center in `init`.

Returns a vector of clusters, where each cluster is a vector of samples.
"""
function buildClusters(xs::AbstractVector{V}, init::AbstractVector{V}, norm::Norm{V})::Vector{Vector{V}} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
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

export KMeans, ClusterInit, UniformRandomInit, KMeansPPInit, CentroidCalculator, EuclideanMeanCentroid, EuclideanNorm

end

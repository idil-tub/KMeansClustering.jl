module KMeansClustering

import Distributions.Distribution
import Distributions.Uniform
import Distributions.rand
import Random.AbstractRNG
import LinearAlgebra.norm
import Statistics.mean

const NonInteger = Core.Real
abstract type ClusterInit{V<:AbstractArray{<:NonInteger}} end

function (c::ClusterInit{V})(samples::AbstractVector{V}, k::Int64)::Vector{V} where {T<:NonInteger,N,V<:AbstractArray{T,N}}
    error("Method initialize not implemented for $(typeof(c))")
end

struct UniformRandomInit{V<:AbstractArray{<:NonInteger}} <: ClusterInit{V} end
function (c::UniformRandomInit{V})(samples::AbstractVector{V}, k::Int64)::Vector{V} where {T<:NonInteger,N,V<:AbstractArray{T,N}}
    dims = size(samples[1])
    min_bounds = fill(typemax(T), dims...)
    max_bounds = fill(typemin(T), dims...)
    for sample in samples
        min_bounds = min.(min_bounds, sample)
        max_bounds = max.(max_bounds, sample)
    end

    function generateSample(min::T, max::T)
        dist = Uniform{T}(min, max)
        return rand(dist)
    end


    return [collect(map(generateSample, min_bounds, max_bounds)) for _ in 1:k]
end

@enum KMeansAlgorithm begin
    Lloyd
end


# partially lifted from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans
function KMeans(x::AbstractVector{V}, k::Int64; init::ClusterInit{V}=UniformRandomInit{V}(), max_iter=300, tol=0.0001, algorithm::KMeansAlgorithm=Lloyd)::Dict{V, Vector{V}} where {T<:NonInteger,N,V<:AbstractArray{T,N}}
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
    return return Dict(zip(centers, clusters))
end

function buildClusters(xs::AbstractVector{V}, init::AbstractVector{V})::Vector{Vector{V}} where {T<:NonInteger,N,V<:AbstractArray{T,N}}
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

function calculateCenter(xs::AbstractVector{V})::V where {T<:NonInteger,N,V<:AbstractArray{T,N}}
    return mean(xs)
end


export KMeans, ClusterInit, UniformRandomInit, buildClusters, calculateCenter

end
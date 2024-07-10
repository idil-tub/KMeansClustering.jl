module KMeansAlgorithms

using ..Types: NonInteger
using ..Norm: NormSqr
using ..Init: ClusterInit
using ..Centroid: CentroidCalculator

"""
    KMeansAlgorithm{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}}

An abstract type representing K-means clustering algorithms.

This type is parameterized by `V`, which can be either a non-integer numeric type or 
an array of non-integer numeric types, representing the type of data points being clustered.

# Type Parameters
- `V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}`: The type of the data points.
  Can be either a non-integer numeric type or an array of non-integer numeric types.

# Purpose
Subtypes of `KMeansAlgorithm` are expected to implement specific variants of the 
K-means clustering algorithm, such as Lloyd's algorithm, mini-batch K-means, or 
other custom methods.

# Examples
```julia
# A concrete subtype might be implemented as:
struct MyAlgorithm{V<:AbstractVector{<:NonInteger}} <: KMeansAlgorithm{V} end
```
See also [`Lloyd`](@ref), [`BkMeans`](@ref)
"""
abstract type KMeansAlgorithm{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} end

"""
    (a::KMeansAlgorithm{V})(samples::AbstractVector{V}, k::Int64, init::ClusterInit{V}, max_iter::Int64, tol::Float64, centroid::CentroidCalculator{V}, normSqr::NormSqr{V})::AbstractVector{Pair{V, AbstractVector{V}}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}

Execute a K-means clustering algorithm on the given samples.

This is an abstract method that should be implemented by concrete subtypes of `KMeansAlgorithm`.
If called on the abstract type, it raises an error.

# Arguments
- `samples::AbstractVector{V}`: Input data samples to be clustered.
- `k::Int64`: Number of clusters to form.
- `init::ClusterInit{V}`: Initialization method for cluster centers.
- `max_iter::Int64`: Maximum number of iterations for the algorithm.
- `tol::Float64`: Convergence tolerance.
- `centroid::CentroidCalculator{V}`: Method to calculate cluster centroids.
- `normSqr::NormSqr{V}`: Squared norm function for distance calculations.

# Returns
- `AbstractVector{Pair{V, AbstractVector{V}}}`: A vector of pairs, where each pair consists of a cluster center and a vector of samples assigned to that center.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input samples and cluster centers.

# Errors
- Throws an error if the method is not implemented for the specific subtype of `KMeansAlgorithm`.

# Example
```julia
# Implementing for a concrete subtype:
function (a::MyCustomKMeans{V})(samples::AbstractVector{V}, k::Int64, init::ClusterInit{V}, max_iter::Int64, tol::Float64, centroid::CentroidCalculator{V}, normSqr::NormSqr{V})::AbstractVector{Pair{V, AbstractVector{V}}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    # Custom K-means algorithm implementation
    return cluster_results
end
```
See also [`KMeansAlgorithm`](@ref), [`Lloyd`](@ref), [`BkMeans`](@ref)
"""
function (a::KMeansAlgorithm{V})(samples::AbstractVector{V}, k::Int64, init::ClusterInit{V}, max_iter::Int64, tol::Float64, centroid::CentroidCalculator{V}, normSqr::NormSqr{V})::AbstractVector{Pair{V, AbstractVector{V}}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    error("Method not implemented for $(typeof(a))")
end

"""
    Lloyd{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} <: KMeansAlgorithm{V}

A concrete implementation of `KMeansAlgorithm` representing Lloyd's algorithm for K-means clustering.

Lloyd's algorithm, also known as the standard K-means algorithm, iteratively reassigns points 
to the nearest cluster center and recalculates cluster centers until convergence or a maximum 
number of iterations is reached.

# Type Parameters
- `V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}`: The type of input samples and cluster centers.
  Can be either a non-integer numeric type or an array of non-integer numeric types.

# Examples
```julia
# Create an instance for vector data
lloyd_kmeans = Lloyd{Vector{Float64}}()
```
See also [`KMeansAlgorithm`](@ref), [`BkMeans`](@ref)
"""
struct Lloyd{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} <: KMeansAlgorithm{V} end

"""
    (a::Lloyd{V})(samples::AbstractVector{V}, k::Int64, init::ClusterInit{V}, max_iter::Int64, tol::Float64, centroid::CentroidCalculator{V}, normSqr::NormSqr{V})::AbstractVector{Pair{V, AbstractVector{V}}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}

Execute Lloyd's K-means clustering algorithm on the given samples.

This method implements the call operator for `Lloyd`, allowing instances to be used as functions
to perform K-means clustering using Lloyd's algorithm.

# Arguments
- `samples::AbstractVector{V}`: Input data samples to be clustered.
- `k::Int64`: Number of clusters to form.
- `init::ClusterInit{V}`: Initialization method for cluster centers.
- `max_iter::Int64`: Maximum number of iterations for the algorithm.
- `tol::Float64`: Convergence tolerance.
- `centroid::CentroidCalculator{V}`: Method to calculate cluster centroids.
- `normSqr::NormSqr{V}`: Squared norm function for distance calculations.

# Returns
- `AbstractVector{Pair{V, AbstractVector{V}}}`: A vector of pairs, where each pair consists of a cluster center and a vector of samples assigned to that center.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input samples and cluster centers.

# Throws
- `ArgumentError`: If `k ≤ 0` or if `k` is greater than the number of samples.

# Examples
```julia
lloyd = Lloyd{Vector{Float64}}()
samples = [rand(3) for _ in 1:100]  # 100 3D points
k = 5
init = KMeansPPInit{Vector{Float64}}()
max_iter = 100
tol = 1e-4
centroid = EuclideanMeanCentroid{Vector{Float64}}()
norm_sqr = EuclideanNormSqr{Vector{Float64}}()
results = lloyd(samples, k, init, max_iter, tol, centroid, norm_sqr)
```
See also [`KMeansAlgorithm`](@ref), [`Lloyd`](@ref), [`BkMeans`](@ref)
"""
function (a::Lloyd{V})(samples::AbstractVector{V}, k::Int64, init::ClusterInit{V}, max_iter::Int64, tol::Float64, centroid::CentroidCalculator{V}, normSqr::NormSqr{V})::AbstractVector{Pair{V, AbstractVector{V}}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    if length(samples) == 0
        return []
    end
    if k <= 0
        throw(ArgumentError("k has to be > 0"))
    end
    if k > length(samples)
        throw(ArgumentError("more clusters than samples"))
    end
    centers = lloyd_intern(samples, init(samples, k, normSqr), max_iter, tol, centroid, normSqr)
    clusters = buildClusters(samples, centers, normSqr)
    return [x => y for (x, y) in zip(centers, clusters)]
end

function lloyd_intern(samples::Vector{V}, centers::Vector{V}, max_iter::Int64, tol::Float64, centroid::CentroidCalculator{V}, normSqr::NormSqr{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    iter = 0
    err = typemax(T)
    clusters = []
    while iter < max_iter && err > tol
        clusters = buildClusters(samples, centers, normSqr)
        new_centers = [length(cluster) != 0 ? centroid(cluster, normSqr) : rand(samples) for cluster in clusters]
        err = sum(normSqr.(centers .- new_centers))
        centers = new_centers
        iter += 1
    end
    return centers
end

"""
    buildClusters(xs::AbstractVector{V}, init::AbstractVector{V}, normSqr::NormSqr{V})::Vector{Vector{V}} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Assigns each sample in `xs` to the nearest cluster center in `init`.

Returns a vector of clusters, where each cluster is a vector of samples.
"""
function buildClusters(xs::AbstractVector{V}, init::AbstractVector{V}, normSqr::NormSqr{V})::Vector{Vector{V}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    num_clusters = length(init)
    clusters = [Vector{V}() for _ in 1:num_clusters]
    for x in xs
        min_dst = typemax(T)
        min_index = 0
        for i in 1:num_clusters
            d = normSqr(x - init[i])
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
    BkMeans{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} <: KMeansAlgorithm{V}

A concrete implementation of `KMeansAlgorithm` representing the BkMeans clustering algorithm.

[BkMeans](https://arxiv.org/abs/2006.15666) is a variant of the K-means algorithm that aims to improve clustering quality
by iteratively refining the solution through a process of "breathing" (adding and removing centers).

# Type Parameters
- `V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}`: The type of input samples and cluster centers.
  Can be either a non-integer numeric type or an array of non-integer numeric types.

# Fields
- `m::Int64`: The number of centers to add and remove in each iteration. Default is 5.
- `eps::Float64`: A small value used in the center perturbation step. Default is 0.001.

# Constructor
    BkMeans{V}(m::Int64 = 5, eps::Float64 = 0.001) where V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}

Constructs a BkMeans instance with the specified parameters.

# Arguments
- `m::Int64`: The number of centers to add and remove in each iteration. Must be non-negative.
- `eps::Float64`: A small value used in the center perturbation step. Must be non-negative.

# Throws
- `ArgumentError`: If either `m` or `eps` is negative.

# Examples
```julia
# Create an instance for vector data with default parameters
bkmeans = BkMeans{Vector{Float64}}()

# Create an instance with custom parameters
bkmeans_custom = BkMeans{Vector{Float64}}(10, 0.0005)
```
"""
struct BkMeans{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} <: KMeansAlgorithm{V} 
    m::Int64
    eps::Float64
    function BkMeans{V}(m::Int64 = 5, eps=0.001) where V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}
        if m < 0 || eps < 0
            throw(ArgumentError("arugments must be non-negative"))
        end
        new(m, eps)
    end
end

"""
    (a::BkMeans{V})(samples::AbstractVector{V}, k::Int64, init::ClusterInit{V}, max_iter::Int64, tol::Float64, centroid::CentroidCalculator{V}, normSqr::NormSqr{V})::AbstractVector{Pair{V, AbstractVector{V}}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}

Execute the BkMeans clustering algorithm on the given samples.

This method implements the call operator for `BkMeans`, allowing instances to be used as functions
to perform K-means clustering using the BkMeans algorithm variant. BkMeans aims to improve clustering
quality by iteratively refining the solution through a process of "breathing" (adding and removing centers).

# Arguments
- `samples::AbstractVector{V}`: Input data samples to be clustered.
- `k::Int64`: Number of clusters to form.
- `init::ClusterInit{V}`: Initialization method for cluster centers.
- `max_iter::Int64`: Maximum number of iterations for the algorithm.
- `tol::Float64`: Convergence tolerance.
- `centroid::CentroidCalculator{V}`: Method to calculate cluster centroids.
- `normSqr::NormSqr{V}`: Squared norm function for distance calculations.

# Returns
- `AbstractVector{Pair{V, AbstractVector{V}}}`: A vector of pairs, where each pair consists of a cluster center and a vector of samples assigned to that center.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input samples and cluster centers.

# Examples
```julia
bkmeans = BkMeans{Vector{Float64}}(5, 0.001)
samples = [rand(3) for _ in 1:100]  # 100 3D points
k = 5
init = KMeansPPInit{Vector{Float64}}()
max_iter = 100
tol = 1e-4
centroid = EuclideanMeanCentroid{Vector{Float64}}()
norm_sqr = EuclideanNormSqr{Vector{Float64}}()
results = bkmeans(samples, k, init, max_iter, tol, centroid, norm_sqr)
```
See also [`KMeansAlgorithm`](@ref), [`Lloyd`](@ref), [`BkMeans`](@ref)
"""
function (a::BkMeans{V})(samples::AbstractVector{V}, k::Int64, init::ClusterInit{V}, max_iter::Int64, tol::Float64, centroid::CentroidCalculator{V}, normSqr::NormSqr{V})::AbstractVector{Pair{V, AbstractVector{V}}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    calc_err = function(centers)
        num_clusters = length(centers)
        err = zero(T)
        for sample in samples
            min_dst = typemax(T)
            for i in 1:num_clusters
                d = normSqr(sample - centers[i])
                if d < min_dst
                    min_dst = d
                end
            end
            err += min_dst
        end
        return err
    end
    calc_cluster_error = function(center, cluster)
        return sum([normSqr(center-el) for el in cluster])
    end
    calc_rmse = function(centers)
        return sqrt(calc_err(centers)/length(samples))
    end
    gen_random_unit = function()
        dist = Uniform{T}(zero(0), one(1))
        return V <: AbstractArray ? rand(dist, sizeof(samples[1])) : rand(dist)
    end
    calc_utilities = function(centers)
        total_error = calc_err(centers)
        utilities = []
        for i in 1:length(centers)
            push!(utilities, calc_err([x for (j, x) in enumerate(centers) if i != j]) - total_error)
        end
        return utilities
        
    end

    c = Lloyd{V}()(samples, k, init, max_iter, tol, centroid, normSqr)
    centers = map(c -> c.first, c)
    clusters = map(c -> c.second, c)

    
    err_best = calc_err(centers)
    c_best = c

    m = min(a.m, k)

    while m > 0
        clusters = buildClusters(samples, centers, normSqr)
        errors = calc_cluster_error.(centers, clusters)
        centers = centers[sortperm(errors, rev=true)]
        rmse = calc_rmse(centers)

        # breath in
        dplus = map((center) -> center + a.eps*rmse * gen_random_unit(), centers[1:m])
        append!(centers, dplus)

        centers = lloyd_intern(samples, centers, max_iter, tol, centroid, normSqr)
        utilities = calc_utilities(centers)
        centers = centers[sortperm(utilities)]

        dminus = Vector{V}()
        f = Vector{V}()

        for c in centers
            if c ∉ f
                push!(dminus, c)
                if length(f) + m < length(centers)
                    c_freeze = argmin(x -> normSqr(x-c ), setdiff(centers, c))
                    push!(f, c_freeze)
                end
                if length(dminus) == m
                    break
                end
            end
        end

        # breath out
        centers = setdiff(centers, dminus)
        centers = lloyd_intern(samples, centers, max_iter, tol, centroid, normSqr)

        error = calc_err(centers)
        if error <= err_best*(1.0-tol)
            c_best = centers
            err_best = error
        else
            m -= 1
        end
        
        m = 0
    end
        
    clusters = buildClusters(samples, c_best, normSqr)
    return [x => y for (x, y) in zip(c_best, clusters)]
end

export KMeansAlgorithm, Lloyd, BkMeans 
    
end
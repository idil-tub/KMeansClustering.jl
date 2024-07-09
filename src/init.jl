module Init
import Distributions.Uniform
import Distributions.Categorical
import Distributions.rand

using ..Types: NonInteger
using ..Norm: NormSqr

"""
    ClusterInit{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}}

An abstract type representing cluster initialization strategies for clustering algorithms.

This type is parameterized by `V`, which can be either a non-integer numeric type or 
an array of non-integer numeric types.

# Purpose
Subtypes of `ClusterInit` are expected to implement specific initialization strategies 
for clustering algorithms, such as k-means++ initialization, random initialization, 
or other custom methods.

# Examples
```julia
# A concrete subtype might be implemented as:
struct MyRandomInit{V<:AbstractVector{<:NonInteger}} <: ClusterInit{V} end
```
See also [`UniformRandomInit`](@ref), [`KMeansPPInit`](@ref)
"""
abstract type ClusterInit{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} end


"""
    (c::ClusterInit{V})(samples::AbstractVector{V}, k::Int64, normSqr::NormSqr{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}

Initialize `k` cluster centers from `samples` using the strategy defined by `c` and the norm `normSqr`.

This is an abstract method that should be implemented by concrete subtypes of `ClusterInit`.
If called on the abstract type, it raises an error.

# Arguments
- `samples::AbstractVector{V}`: Input data samples.
- `k::Int64`: Number of cluster centers to initialize.
- `normSqr::NormSqr{V}`: Norm function to be used in the initialization process.

# Returns
- `Vector{V}`: A vector of `k` initialized cluster centers.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input samples and cluster centers.

# Errors
- Throws an error if the method is not implemented for the specific subtype of `ClusterInit`.

# Example
```julia
# Implementing for a concrete subtype:
function (c::MyCustomInit{V})(samples::AbstractVector{V}, k::Int64, normSqr::NormSqr{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    # Custom initialization logic here
    return initialize_centers(samples, k, normSqr)
end
```
See also [`ClusterInit`](@ref) [`UniformRandomInit`](@ref), [`KMeansPPInit`](@ref)
end
"""
function (c::ClusterInit{V})(samples::AbstractVector{V}, k::Int64, normSqr::NormSqr{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    error("Method not implemented for $(typeof(c))")
end

"""
    UniformRandomInit{V<:Union{AbstractArray{<:NonInteger},<:NonInteger}} <: ClusterInit{V}

A concrete implementation of `ClusterInit` representing uniform random initialization for clustering algorithms.

This struct implements the uniform random initialization strategy, where initial cluster centers
are chosen uniformly at random from the bounding hyperrectangle of the input samples.

# Type Parameters
- `V<:Union{AbstractArray{<:NonInteger},<:NonInteger}`: The type of input samples and cluster centers.
  Can be either a non-integer numeric type or an array of non-integer numeric types.

# Examples
```julia
# Create an instance for vector data
uniform_init = UniformRandomInit{Vector{Float64}}()
```
See also [`ClusterInit`](@ref), [`KMeansPPInit`](@ref)
"""
struct UniformRandomInit{V<:Union{AbstractArray{<:NonInteger},<:NonInteger}} <: ClusterInit{V} end

"""
    (c::UniformRandomInit{V})(samples::AbstractVector{V}, k::Int64, normSqr::NormSqr{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}

Initialize `k` cluster centers from `samples` using uniform random distribution over the bounding hyperrectangle of the samples.

This method implements the call operator for `UniformRandomInit`, allowing instances
to be used as functions to generate initial cluster centers.

# Arguments
- `samples::AbstractVector{V}`: Input data samples.
- `k::Int64`: Number of cluster centers to initialize.
- `normSqr::NormSqr{V}`: Norm function (not used in this implementation but required for interface consistency).

# Returns
- `Vector{V}`: A vector of `k` initialized cluster centers.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input samples and cluster centers.

# Examples
```julia
uniform_init = UniformRandomInit{Vector{Float64}}()
samples = [rand(3) for _ in 1:100]  # 100 3D points
k = 5
norm_sqr = EuclideanNormSqr{Vector{Float64}}()
centers = uniform_init(samples, k, norm_sqr)
```
See also [`ClusterInit`](@ref) [`UniformRandomInit`](@ref), [`KMeansPPInit`](@ref)
"""
function (c::UniformRandomInit{V})(samples::AbstractVector{V}, k::Int64, normSqr::NormSqr{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
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

"""
    KMeansPPInit{V<:Union{AbstractArray{<:NonInteger},<:NonInteger}} <: ClusterInit{V}

A concrete implementation of `ClusterInit` representing the k-means++ initialization strategy for clustering algorithms.

This struct implements the k-means++ initialization method, which selects initial cluster centers
with a probability proportional to their squared distance from the closest center already chosen.

# Type Parameters
- `V<:Union{AbstractArray{<:NonInteger},<:NonInteger}`: The type of input samples and cluster centers.
  Can be either a non-integer numeric type or an array of non-integer numeric types.

# Examples
```julia
# Create an instance for vector data
kmeans_pp_init = KMeansPPInit{Vector{Float64}}()
```
See also [`ClusterInit`](@ref) [`UniformRandomInit`](@ref)
"""
struct KMeansPPInit{V<:Union{AbstractArray{<:NonInteger},<:NonInteger}} <: ClusterInit{V} end

"""
    (c::KMeansPPInit{V})(samples::AbstractVector{V}, k::Int64, normSqr::NormSqr{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Perform K-means++ initialization to select initial cluster centers.

This method implements the call operator for `KMeansPPInit`, allowing instances
to be used as functions to generate initial cluster centers using the K-means++
algorithm.

# Arguments
- `samples::AbstractVector{V}`: Input data samples.
- `k::Int64`: Number of cluster centers to initialize.
- `normSqr::NormSqr{V}`: Norm function used to calculate distances between points.

# Returns
- `Vector{V}`: A vector of `k` initialized cluster centers.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input samples and cluster centers.

# Examples
```julia
kmeans_pp = KMeansPPInit{Vector{Float64}}()
samples = [rand(3) for _ in 1:100]  # 100 3D points
k = 5
norm_sqr = EuclideanNormSqr{Vector{Float64}}()
centers = kmeans_pp(samples, k, norm_sqr)
```
See also [`ClusterInit`](@ref) [`UniformRandomInit`](@ref), [`KMeansPPInit`](@ref)
"""
function (c::KMeansPPInit{V})(samples::AbstractVector{V}, k::Int64, normSqr::NormSqr{V})::Vector{V} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    centers = []
    n = length(samples)
    probabilities = fill(1 / n, n)
    while length(centers) < k
        push!(centers, samples[rand(Categorical(probabilities))])
        total_distance = zero(T)
        for (i, sample) in enumerate(samples)
            min_dist_sq = Inf
            for center in centers
                dist_sq = normSqr(sample - center)^2
                min_dist_sq = min(min_dist_sq, dist_sq)
            end
            probabilities[i] = min_dist_sq
            total_distance += min_dist_sq
        end
        probabilities /= total_distance
    end
    return centers
end

export ClusterInit, UniformRandomInit, KMeansPPInit

end
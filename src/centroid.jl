"""
    KMeansClustering.Centroid

This module provides abstract and concrete types for centroid calculation.

# Types
- `CentroidCalculator{V}`: Abstract type for centroid calculators.
- `EuclideanMeanCentroid{V}`: Concrete type for Euclidean mean centroid calculation.

# Functions
- Call operators for `CentroidCalculator` and `EuclideanMeanCentroid` to perform centroid calculations

# Exports
- `CentroidCalculator`
- `EuclideanMeanCentroid`
"""
module Centroid

import Statistics.mean

using ..Types: NonInteger
using ..Norm: NormSqr
"""
    CentroidCalculator{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}}

An abstract type representing methods for calculating centroids in clustering algorithms.

This type is parameterized by `V`, which can be either a non-integer numeric type or 
an array of non-integer numeric types, representing the type of data points for which 
centroids are being calculated.

# Type Parameters
- `V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}`: The type of the data points.
  Can be either a non-integer numeric type or an array of non-integer numeric types.

# Purpose
Subtypes of `CentroidCalculator` can be used to implement different ways of calculating the centroid for k-Means.

# Examples
```julia
# A concrete subtype might be implemented as:
struct CustomCentroid{V<:AbstractVector{<:NonInteger}} <: CentroidCalculator{V} end
```
See also [`EuclideanMeanCentroid`](@ref)
"""
abstract type CentroidCalculator{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} end
"""
    (c::CentroidCalculator{V})(samples::AbstractVector{V}, normSqr::NormSqr{V})::V where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Calculate the center of a cluster from the given `samples` using the provided `normSqr`.

This is an abstract method that should be implemented by concrete subtypes of `CentroidCalculator`.
If called on the abstract type, it raises an error.

# Arguments
- `samples::AbstractVector{V}`: A vector of data points in the cluster.
- `normSqr::NormSqr{V}`: The squared norm function used for distance calculations.

# Returns
- `V`: The calculated center of the cluster.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input samples and the returned center.

# Errors
- Throws an error if the method is not implemented for the specific subtype of `CentroidCalculator`.

# Example
```julia
function (c::CustomCentroid{V})(samples::AbstractVector{V}, normSqr::NormSqr{V})::V where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}
    return samples[1]
end
```
See also [`CentroidCalculator`](@ref), [`EuclideanMeanCentroid`](@ref)
"""
function (c::CentroidCalculator{V})(samples::AbstractVector{V}, normSqr::NormSqr{V})::V where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    error("Method not implemented for $(typeof(c))")
end

"""
    EuclideanMeanCentroid{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} <: CentroidCalculator{V}

A concrete implementation of `CentroidCalculator` that calculates the centroid of a cluster 
using the Euclidean mean (arithmetic average) of the sample points.

This centroid calculator is suitable for use with Euclidean distance-based clustering algorithms.

# Type Parameters
- `V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}`: The type of input samples and calculated centroids.
  Can be either a non-integer numeric type or an array of non-integer numeric types.

# Examples
```julia
# Create an instance for vector data
euclidean_mean = EuclideanMeanCentroid{Vector{Float64}}()
```
See also [`CentroidCalculator`](@ref)
"""
struct EuclideanMeanCentroid{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} <: CentroidCalculator{V} end

"""
    (c::EuclideanMeanCentroid{V})(samples::AbstractVector{V}, normSqr::NormSqr{V})::V where {T<:NonInteger,N,V<:Union{T, AbstractArray{T,N}}}

Calculate the centroid of a cluster using the standard Euclidean mean (arithmetic average) of the sample points.

This method implements the call operator for `EuclideanMeanCentroid`, allowing instances
to be used as functions to compute cluster centroids.

# Arguments
- `samples::AbstractVector{V}`: A vector of data points in the cluster.
- `normSqr::NormSqr{V}`: The squared norm function (not used in this implementation but required for interface consistency).

# Returns
- `V`: The calculated centroid of the cluster, which is the arithmetic mean of all points.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input samples and the returned centroid.

# Examples
```julia
euclidean_mean = EuclideanMeanCentroid{Vector{Float64}}()
samples = [rand(3) for _ in 1:10]  # 10 3D points
norm_sqr = EuclideanNormSqr{Vector{Float64}}()
centroid = euclidean_mean(samples, norm_sqr)
```
See also [`CentroidCalculator`](@ref), [`EuclideanMeanCentroid`](@ref)
"""
function (c::EuclideanMeanCentroid{V})(samples::AbstractVector{V}, normSqr::NormSqr{V})::V where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    return mean(samples)
end

export CentroidCalculator, EuclideanMeanCentroid

end
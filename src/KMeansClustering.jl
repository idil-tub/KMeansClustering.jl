module KMeansClustering

include("types.jl")
include("norm.jl")
include("init.jl")
include("centroid.jl")
include("kmeans_algorithms.jl")

using .Types
using .Norm
using .Init
using .Centroid
using .KMeansAlgorithms


"""
    KMeans(x::AbstractVector{V}, k::Int64; init::ClusterInit{V}=UniformRandomInit{V}(), max_iter::Int64=300, tol::Float64=0.0001, algorithm::KMeansAlgorithm=Lloyd{V}(), centroid::CentroidCalculator{V}=EuclideanMeanCentroid{V}(), normSqr::NormSqr{V}=EuclideanNormSqr{V}())::AbstractVector{Pair{V, AbstractVector{V}}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}

Perform K-means clustering on the data `x` with `k` clusters.

This function provides a flexible interface to various K-means clustering implementations,
allowing customization of initialization, algorithm, centroid calculation, and distance metric.

# Arguments
- `x::AbstractVector{V}`: Input data as an abstract vector of type `V`.
- `k::Int64`: Number of clusters to form.

# Keyword Arguments
- `init::ClusterInit{V}=UniformRandomInit{V}()`: Cluster initialization method.
- `max_iter::Int64=300`: Maximum number of iterations.
- `tol::Float64=0.0001`: Tolerance for convergence.
- `algorithm::KMeansAlgorithm=Lloyd{V}()`: K-means algorithm variant to use.
- `centroid::CentroidCalculator{V}=EuclideanMeanCentroid{V}()`: Method to calculate cluster centroids.
- `normSqr::NormSqr{V}=EuclideanNormSqr{V}()`: Squared norm function for distance calculations.

# Returns
- `AbstractVector{Pair{V, AbstractVector{V}}}`: A vector of pairs, where each pair consists of a cluster center and a vector of samples assigned to that center.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input samples and cluster centers.

# Examples
```julia
# Basic usage with default parameters
data = [rand(3) for _ in 1:100]  # 100 3D points
result = KMeans(data, 5)

# Custom configuration
result = KMeans(data, 5, 
    init=KMeansPPInit{Vector{Float64}}(),
    max_iter=500,
    tol=1e-6,
    algorithm=BkMeans{Vector{Float64}}(10, 0.001),
    centroid=EuclideanMeanCentroid{Vector{Float64}}(),
    normSqr=EuclideanNormSqr{Vector{Float64}}()
)
```
See also [`ClusterInit`](@ref), [`KMeansAlgorithm`](@ref), [`CentroidCalculator`](@ref), [`NormSqr`](@ref)
"""
function KMeans(
    x::AbstractVector{V}, k::Int64; 
    init::ClusterInit{V}=UniformRandomInit{V}(), 
    max_iter::Int64=300, tol::Float64=0.0001, 
    algorithm::KMeansAlgorithm=Lloyd{V}(), 
    centroid::CentroidCalculator{V}=EuclideanMeanCentroid{V}(), 
    normSqr::NormSqr{V}=EuclideanNormSqr{V}())::AbstractVector{Pair{V, AbstractVector{V}}} where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    return algorithm(x, k, init, max_iter, tol, centroid, normSqr)
end

export NonInteger, NormSqr, EuclideanNormSqr, ClusterInit, UniformRandomInit, KMeansPPInit, CentroidCalculator, EuclideanMeanCentroid, KMeansAlgorithm, Lloyd, BkMeans, KMeans

end

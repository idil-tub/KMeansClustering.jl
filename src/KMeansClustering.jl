module KMeansClustering

import Distributions.Distribution
import Distributions.Uniform
import Random.AbstractRNG

const NonInteger = Union{Rational, AbstractFloat}
abstract type ClusterInit{T<:NonInteger, N} end

function (c::ClusterInit{T})(samples::AbstractVector{AbstractArray{T, N}}, k::Int64)::AbstractArray{T, N} where {T<:NonInteger, N}
    error("Method initialize not implemented for $(typeof(c))")
end


struct RandomInit{T, N} <: ClusterInit{T, N}
    # el type should match T, size should be 1 or match the sample
    distribution::Distribution
    rng::AbstractRNG
end

function (c::RandomInit{T})(samples::AbstractVector{AbstractArray{T, N}}, k::Int64)::AbstractArray{T, N} where {T<:NonInteger, N}
    missing
end

struct RandomFromSamplesInit{T, N} <: ClusterInit{T, N}
    # el type should match T, size should be 1 or match the sample
    distribution::Function
    rng::AbstractRNG
end

function (c::RandomInit{T})(samples::AbstractVector{AbstractArray{T, N}}, k::Int64)::AbstractArray{T, N} where {T<:NonInteger, N}
    missing
end

# # TODO: more initializer types, specifically kmeans++
# # In general I'm not sure about this design. Maybe just a closure is enough?



# partially lifted from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans
function KMeans(x::AbstractVector{AbstractArray{T, N}}, k::Int64; init=ClusterInit, max_iter=300, tol=0.0001, algorithm=Union{"lloyd"}) where {T<:NonInteger, N}
    missing
end

export KMeans, ClusterInit, RandomFromSamplesInit, RandomInit

end
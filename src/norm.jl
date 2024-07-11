"""
    KMeansClustering.Norm

This module defines abstract and concrete types for calculating norms, particularly
squared norms, which are essential in many clustering and distance-based algorithms.

# Types
- `NormSqr{V}`: Abstract type for squared norm calculations
- `EuclideanNormSqr{V}`: Concrete type for squared Euclidean norm

# Functions
- Call operators for `NormSqr` and `EuclideanNormSqr` to perform norm calculations

# Exports
- `NormSqr`
- `EuclideanNormSqr`
"""
module Norm
import LinearAlgebra.norm_sqr as la_norm_sqr

using ..Types: NonInteger

"""
    NormSqr{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}}

An abstract type representing squared norms.

The type parameter `V` can be either a non-integer numeric type or an array of non-integer numeric types.
This allows for representation of squared norms of scalars, vectors, or matrices with non-integer elements.

# Example
```julia
# Concrete subtypes might be implemented as:
# struct MyNormSqr{V<:AbstractVector{<:NonInteger}} <: NormSqr{V}
#     myParameter::Any
# end
See also: [`EuclideanNormSqr`](@ref)
"""
abstract type NormSqr{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} end

"""
    (c::NormSqr{V})(x::V)::T where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}

Compute the squared norm of `x` using the norm represented by `c`.

# Returns
- `T`: The computed squared norm, which is a non-integer numeric type.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements and the result.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input, either a scalar or an array.

This is an abstract method that should be implemented by concrete subtypes of `NormSqr`.
If called on the abstract type, it raises an error.

# Errors
- Throws an error if the method is not implemented for the specific subtype of `NormSqr`.

# Example
```julia
function (c::MyNormSqr{V})(x::V)::T where V<:AbstractVector{<:NonInteger}
    return sum(abs, x)
end
```
"""
function (c::NormSqr{V})(x::V)::T where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    error("Method not implemented for $(typeof(c))")
end

"""
    EuclideanNormSqr{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} <: NormSqr{V}

A concrete implementation of `NormSqr` representing the squared Euclidean norm.

# Type Parameters
- `V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}`: The type of input for which this norm is defined.

# Example
```julia
# Create an instance
euclidean_norm_sqr = EuclideanNormSqr{Vector{Float64}}()
```
See also: [`NormSqr`](@ref)
"""
struct EuclideanNormSqr{V<:Union{<:NonInteger,AbstractArray{<:NonInteger}}} <: NormSqr{V} end

"""
    (c::EuclideanNormSqr{V})(x::V)::T where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}

Compute the squared Euclidean norm of `x`.

This method implements the call operator for `EuclideanNormSqr`, allowing instances
to be used as functions to calculate squared Euclidean norms.

# Returns
- `T`: The computed squared Euclidean norm, which is a non-integer numeric type.

# Type Parameters
- `T<:NonInteger`: The numeric type of the elements and the result.
- `N`: The number of dimensions if `V` is an array type.
- `V<:Union{T,AbstractArray{T,N}}`: The type of the input, either a scalar or an array.

# Example
```julia
euclidean_norm_sqr = EuclideanNormSqr{Vector{Float64}}()
v = [3.0, 4.0]
result = euclidean_norm_sqr(v)  # Returns 25.0
```
See also: [`NormSqr`](@ref)
"""
function (c::EuclideanNormSqr{V})(x::V)::T where {T<:NonInteger,N,V<:Union{T,AbstractArray{T,N}}}
    return la_norm_sqr(x)
end

export NormSqr, EuclideanNormSqr

end
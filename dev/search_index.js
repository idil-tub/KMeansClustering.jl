var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = KMeansClustering","category":"page"},{"location":"#KMeansClustering","page":"Home","title":"KMeansClustering","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for KMeansClustering.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [KMeansClustering]","category":"page"},{"location":"#KMeansClustering.ClusterInit-Union{Tuple{V}, Tuple{N}, Tuple{T}, Tuple{AbstractVector{V}, Int64}} where {T<:Real, N, V<:Union{AbstractArray{T, N}, T}}","page":"Home","title":"KMeansClustering.ClusterInit","text":"initialize(c::ClusterInit{V}, samples::AbstractVector{V}, k::Int64)::Vector{V}\n\nInitializes k cluster centers from samples using the cluster initialization method c.\n\n\n\n\n\n","category":"method"},{"location":"#KMeansClustering.UniformRandomInit","page":"Home","title":"KMeansClustering.UniformRandomInit","text":"initialize(c::UniformRandomInit{V}, samples::AbstractVector{V}, k::Int64)::Vector{V}\n\nInitializes k cluster centers from samples using a uniform random distribution.\n\n\n\n\n\n","category":"type"},{"location":"#KMeansClustering.KMeans-Union{Tuple{V}, Tuple{N}, Tuple{T}, Tuple{AbstractVector{V}, Int64}} where {T<:Real, N, V<:Union{AbstractArray{T, N}, T}}","page":"Home","title":"KMeansClustering.KMeans","text":"KMeans(x::AbstractVector{V}, k::Int64; init::ClusterInit{V}=UniformRandomInit{V}(), max_iter=300, tol=0.0001, algorithm::KMeansAlgorithm=Lloyd)::Dict{V, Vector{V}}\n\nPerform K-means clustering on the data x with k clusters.\n\nArguments:\n\nx: Input data as an abstract vector of type V.\nk: Number of clusters.\ninit: Cluster initialization method. Default is UniformRandomInit.\nmax_iter: Maximum number of iterations. Default is 300.\ntol: Tolerance for convergence. Default is 0.0001.\nalgorithm: K-means algorithm to use. Default is Lloyd.\n\nReturns a dictionary mapping each cluster center to its assigned samples.\n\n\n\n\n\n","category":"method"},{"location":"#KMeansClustering.buildClusters-Union{Tuple{V}, Tuple{N}, Tuple{T}, Tuple{AbstractVector{V}, AbstractVector{V}}} where {T<:Real, N, V<:Union{AbstractArray{T, N}, T}}","page":"Home","title":"KMeansClustering.buildClusters","text":"buildClusters(xs::AbstractVector{V}, init::AbstractVector{V})::Vector{Vector{V}}\n\nAssigns each sample in xs to the nearest cluster center in init.\n\nReturns a vector of clusters, where each cluster is a vector of samples.\n\n\n\n\n\n","category":"method"},{"location":"#KMeansClustering.calculateCenter-Union{Tuple{AbstractVector{V}}, Tuple{V}, Tuple{N}, Tuple{T}} where {T<:Real, N, V<:Union{AbstractArray{T, N}, T}}","page":"Home","title":"KMeansClustering.calculateCenter","text":"calculateCenter(xs::AbstractVector{V})::V\n\nCalculates the center of the cluster xs.\n\nReturns the calculated center.\n\n\n\n\n\n","category":"method"}]
}

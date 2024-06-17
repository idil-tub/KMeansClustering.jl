# KMeansClustering

[![Build Status](https://github.com/idil-tub/KMeansClustering.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/idil-tub/KMeansClustering.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Installation



## Usage

```julia
# Generate some sample data
data = rand(2, 100)  # 100 data points in 2 dimensions

# Convert data to an AbstractVector of Vector{Float64}
data_vec = [data[:, i] for i in 1:size(data, 2)]

# Perform k-means clustering
k = 3
max_iter = 100
tol = 0.001
clusters = KMeans(data_vec, k; max_iter=max_iter, tol=tol)

# Print cluster centers and their members
for (center, members) in clusters
    println("Cluster center: ", center)
    println("Members: ", members)
end

```

## Contributing



## License

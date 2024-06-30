# KMeansClustering

[![Build Status](https://github.com/idil-tub/KMeansClustering.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/idil-tub/KMeansClustering.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![Code Coverage](https://codecov.io/github/idil-tub/KMeansClustering.jl/graph/badge.svg?token=MVVT1D4HSD)](https://codecov.io/github/idil-tub/KMeansClustering.jl)

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://idil-tub.github.io/KMeansClustering.jl/dev/)


The KMeansClustering package provides an implementation of the K-means clustering algorithm, allowing for the partitioning of a dataset into k clusters. It includes customizable initialization methods for cluster centers and supports different K-means algorithms.

## Installation

You can install `KMeansClustering.jl` by adding it directly from our GitHub repository. Here are the steps:

1. Open Julia's REPL (the Julia command-line interface).

2. Press `]` to enter Pkg mode (the prompt should change to `pkg>`).

3. Run the following command to add `KMeansClustering.jl`:

```julia
pkg> add https://github.com/idil-tub/KMeansClustering.jl.git
```

 4. Once installed, you can import the package and start using it.
```julia
using KMeansClustering
```

## Usage

```julia
# Generate some sample data
data = rand(2, 100)  # 100 data points in 2 dimensions

# Convert data to an AbstractVector of Vector{Float64}
data_vec = [data[:, i] for i in 1:size(data, 2)]

# Perform k-means clustering
k = 3
max_iter = 100
tol = 0.0001
clusters = KMeans(data_vec, k; max_iter=max_iter, tol=tol)

# Print cluster centers and their members
for (center, members) in clusters
    println("Cluster center: ", center)
    println("Members: ", members)
end

```





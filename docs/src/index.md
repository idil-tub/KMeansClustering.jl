```@meta
CurrentModule = KMeansClustering
```

# KMeansClustering

Documentation for [KMeansClustering](https://github.com/idil-tub/KMeansClustering.jl).


```@index
```

```@autodocs
Modules = [KMeansClustering]
```

## Two Dimensions (Sepal Length and Width) Example of Iris Data

```@example 1
using KMeansClustering
using MLJ
using DataFrames
using Plots

ENV["GKSwstype"] = "100" # hide
gr() # hide

# Load Iris data
models() 
data = load_iris()
iris = DataFrame(data)
y_iris, X_iris = unpack(iris, ==(:target); rng=123);

# Extract sepal_length and sepal_width features from Iris dataset
# Convert selected features to a vector for KMeans clustering
X_iris = Matrix(X_iris)'
X_iris_vec = [Vector{Float64}(col) for col in eachcol(X_iris[1:2, :])]

# Execute KMeans clustering
k = 3
max_iter = 100
tol = 0.0001
clusters = KMeans(X_iris_vec, k; max_iter=max_iter, tol=tol)
display(clusters)
```
```@example 1
# Plot the result
p = plot(title="Two Dimension KMean - Iris", xlabel="sepal_length", ylabel="sepal_width", legend=:topright)
color_palette = palette(:tab10)

for (i, (centers, members)) in enumerate(clusters)

    mem_x = [members[i][1] for i in 1:length(members)]
    mem_y = [members[i][2] for i in 1:length(members)]
    
    # Plot cluster points
    scatter!(p, mem_x, mem_y, label="Cluster $i", color=color_palette[i])
    
    # Plot cluster center
    scatter!(p, [centers[1]], [centers[2]], color=color_palette[i], marker=:star, markersize=10, label="Center $i")

end

display(p)
```

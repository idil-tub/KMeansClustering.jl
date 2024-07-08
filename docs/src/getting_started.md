## Two Dimensions (Sepal Length and Width) Example of Iris Data

This is a simple example with real world data on how to use [KMeans](https://en.wikipedia.org/wiki/K-means_clustering) Clustering and visualize the clusters.

The process consists of four main steps:
1. Data Loading - Iris Data
2. Data Preprocessing - Extract required data and convert to required data format
3. KMeans Clustering Execution
4. Results Visualization - 2D Plot

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
clusters # hide
```


### Visualization

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
savefig(p, "two_dim_kmeans_iris.svg"); nothing # hide
```
![](two_dim_kmeans_iris.svg)

## High Dimensions Example of Wine Data

This example illustrates the application of KMeans Clustering to high-dimensional, real-world data, demonstrating how the algorithm partitions and visualizes complex datasets into distinct clusters.

The process consists of four main steps:
1. Data Loading - Wine Data
2. Data Preprocessing - Normalize and convert to required data format
3. KMeans Clustering Execution
4. Results Visualization - t-SNE to 2D Plot

```@example 2
using KMeansClustering
using TSne
using Random
using HTTP
using CSV
using DataFrames
using MLJ
using Plots
ENV["GKSwstype"] = "100" # hide
gr() # hide

# Load wine data
# There are 13 features: Alcohol,Malic.acid,Ash,Acl,Mg,Phenols,Flavanoids,Nonflavanoid.phenols,Proanth,Color.int,Hue,OD,Proline
data_path = ".\\wine.csv"
url = "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv"
HTTP.download(url, data_path)

wine_df = CSV.read(data_path, DataFrame)
y_wine, X_wine = unpack(wine_df, ==(:Wine); rng=123);

X_wine = Matrix(X_wine)'

# Normalize
X_normalized = (X_wine .- mean(X_wine, dims=1)) ./ std(X_wine, dims=1)

# Convert to vector
X_normalized_vec = [Vector{Float64}(col) for col in eachcol(X_normalized)]

# Execute KMeans clustering
k = 3
max_iter = 100
tol = 0.0001
clusters = KMeans(X_normalized_vec, k; max_iter=max_iter, tol=tol)
clusters # hide
```

### Visualization

We use t-SNE to do dimension reduction for visulization. 

[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) (t-Distributed Stochastic Neighbor Embedding) is a powerful dimensionality reduction and data visualization technique used in machine learning and data science.

```@example 2
# Extract and combine centers and clusters for the purpose of using t-SNE
combined_centers_clusters = Matrix{Float64}(undef, 0, length(first(first(clusters)[2])))
centers_index = Int[]
global count_index = 1
for (i, (centers, members)) in enumerate(clusters)
    global combined_centers_clusters = vcat(combined_centers_clusters, centers')
    push!(centers_index, count_index)
    
    for member in members
        global combined_centers_clusters = vcat(combined_centers_clusters, member')
        global count_index += 1
    end
    
    global count_index += 1
end

# Execute t-SNE
tsne_result = tsne(combined_centers_clusters, 2, 50, 1000, 20.0);

# Convert to vector
tsne_result_vec = [Vector{Float64}(col) for col in eachcol(tsne_result')]

# Plot the result
p = plot(title="High Dimension KMean - Wine", xlabel="t-SNE 1", ylabel="t-SNE 2", legend=:topright)
color_palette = palette(:tab10)

for i in 1:k
    tsne_result_members = []
     
    for j in (centers_index[i]+1):(i != k ? (centers_index[i+1]-1) : length(tsne_result_vec) )
        push!(tsne_result_members, tsne_result_vec[j])
    end

    # Plot cluster points
    mem_x = [tsne_result_members[m][1] for m in 1:length(tsne_result_members)]
    mem_y = [tsne_result_members[m][2] for m in 1:length(tsne_result_members)]

    scatter!(p, mem_x, mem_y, color=color_palette[i] , label="Cluster $i")
    
    # Plot cluster center
    scatter!(p, ([tsne_result_vec[centers_index[i]]][1][1], [tsne_result_vec[centers_index[i]]][1][2]), color=color_palette[i], marker=:star, markersize=10, label="Center $i")
end

savefig(p, "high_dim_kmeans_wine.svg"); nothing # hide
```
![](high_dim_kmeans_wine.svg)

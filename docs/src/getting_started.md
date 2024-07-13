## Two Dimensions (Sepal Length and Width) Example of Iris Data

This is a simple example with real world data on how to use [KMeans](https://en.wikipedia.org/wiki/K-means_clustering) Clustering and visualize the clusters.

The process consists of four main steps:
1. Data Loading - Iris Data
2. Data Preprocessing - Extract required data and convert to required data format
3. KMeans Clustering Execution - KMean and KMean++
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
X_iris = Matrix(X_iris[:, 1:2])'
X_iris_vec = [Vector{Float64}(col) for col in eachcol(X_iris)]

# Execute KMeans clustering
k = 3
clusters = KMeans(X_iris_vec, k,
    init=KMeansPPInit{Vector{Float64}}(),
    max_iter=400,
    tol=0.001,
    algorithm=BkMeans{Vector{Float64}}(5, 0.001),
    centroid=EuclideanMeanCentroid{Vector{Float64}}(),
    normSqr=EuclideanNormSqr{Vector{Float64}}()
)
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
3. KMeans Clustering Execution - KMean and KMean++
4. Results Visualization - t-SNE to 2D Plot

```@example 2
using KMeansClustering
using TSne
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
clusters = KMeans(X_normalized_vec, k,
    init=KMeansPPInit{Vector{Float64}}(),
    max_iter=400,
    tol=0.001,
    algorithm=BkMeans{Vector{Float64}}(5, 0.001),
    centroid=EuclideanMeanCentroid{Vector{Float64}}(),
    normSqr=EuclideanNormSqr{Vector{Float64}}()
)
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

## Comparison of KMean, KMean++ and BKMean

Here, we present a comparative analysis demonstrating the effectiveness of KMeans++ and BKMeans algorithms using the GMD5X5 dataset. We will showcase visualizations of the results and the SSE (Sum of Squared Errors) values obtained.

[SSE](https://en.wikipedia.org/wiki/Residual_sum_of_squares)(Sum of Squared Errors) is a measure used to evaluate the performance of clustering algorithms. It calculates the sum of the squared distances between each data point and the centroid of its assigned cluster. Lower SSE values indicate better clustering as they reflect tighter and more compact clusters.

```@example 3
using HTTP # hide
using CSV # hide
using DataFrames # hide
using KMeansClustering # hide
using Plots # hide

data_path = ".\\gmd5x5.csv" # hide
url = "https://raw.githubusercontent.com/gittar/breathing-k-means/master/data/gmd5x5.csv" # hide
HTTP.download(url, data_path) # hide

gmd55_df = CSV.read(data_path, DataFrame; header=false) # hide

# Convert to vector  # hide
gmd55_df_vec = [Vector{Float64}(row) for row in eachrow(gmd55_df)] # hide

# Execute KMeans clustering  # hide
k = 50 # hide
clusters= KMeans(gmd55_df_vec, k, # hide
    max_iter=400, # hide
    tol=0.001, # hide
    centroid=EuclideanMeanCentroid{Vector{Float64}}(), # hide
    normSqr=EuclideanNormSqr{Vector{Float64}}() # hide
) # hide

clusters_pp = KMeans(gmd55_df_vec, k, # hide
    init=KMeansPPInit{Vector{Float64}}(), # hide
    max_iter=400, # hide
    tol=0.001, # hide
    centroid=EuclideanMeanCentroid{Vector{Float64}}(), # hide
    normSqr=EuclideanNormSqr{Vector{Float64}}() # hide
) # hide

clusters_pp_b = KMeans(gmd55_df_vec, k, # hide
    init=KMeansPPInit{Vector{Float64}}(), # hide
    max_iter=400, # hide
    tol=0.001, # hide
    algorithm=BkMeans{Vector{Float64}}(5, 0.001), # hide
    centroid=EuclideanMeanCentroid{Vector{Float64}}(), # hide
    normSqr=EuclideanNormSqr{Vector{Float64}}() # hide
) # hide

function plot_result(clusters, title, filename) # hide
    # Extract and combine centers and clusters for the purpose of using t-SNE # hide
    combined_centers_clusters = Matrix{Float64}(undef, 0, length(first(first(clusters)[2]))) # hide
    centers_index = Int[] # hide
    clusters_assignments = Int[] # hide
    count_index = 1 # hide
    sse = 0.0 # hide
    rng = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # hide
    count_center = zeros(5, 5) # hide
    for (i, (centers, members)) in enumerate(clusters) # hide

        combined_centers_clusters = vcat(combined_centers_clusters, centers') # hide
        push!(centers_index, count_index) # hide

        xi = centers[1] # hide
        yi = centers[2] # hide
        for m in 1:length(rng)-1 # hide
            for n in 1:length(rng)-1 # hide
                if (rng[n] < xi < rng[n+1]) && (rng[m] < yi < rng[m+1]) # hide
                    count_center[length(rng)-m, n] += 1 # hide
                end # hide
            end # hide
        end # hide
        
        for member in members # hide
            combined_centers_clusters = vcat(combined_centers_clusters, member') # hide
            push!(clusters_assignments, i) # hide
            count_index += 1 # hide
            sse += sum((member .- centers).^ 2) # hide
        end # hide

        count_index += 1 # hide
    end # hide

    combined_centers_clusters = [Vector{Float64}(col) for col in eachcol(combined_centers_clusters')] # hide

    p = plot(title="$title \n SSE:$sse", legend=:topright) # hide
    global result_centers = Matrix{Float64}(undef, 0, 2) # hide
    global result_members_matrix = Matrix{Float64}(undef, 0, 2) # hide

    for i in 1:k # hide
        result_members = [] # hide
        
        result_centers = vcat(result_centers, combined_centers_clusters[centers_index[i]]') # hide

        for j in (centers_index[i]+1):(i != k ? (centers_index[i+1]-1) : length(combined_centers_clusters) ) # hide
           
            push!(result_members, combined_centers_clusters[j]) # hide
            result_members_matrix = vcat(result_members_matrix, combined_centers_clusters[j]') # hide
        end # hide
        
        # Plot cluster points # hide
        mem_x = [result_members[m][1] for m in 1:length(result_members)] # hide
        mem_y = [result_members[m][2] for m in 1:length(result_members)] # hide

        scatter!(p, mem_x, mem_y, color=RGB(0.470588, 0.776471, 0.474510) , markersize=3, label=false, markerstrokewidth = 0.5) # hide

        # Plot cluster center # hide
        for m in 1:length(rng)-1 # hide
            for n in 1:length(rng)-1 # hide
                xi = [combined_centers_clusters[centers_index[i]]][1][1] # hide
                yi = [combined_centers_clusters[centers_index[i]]][1][2] # hide
                if (rng[n] < xi < rng[n+1]) && (rng[m] < yi < rng[m+1]) && count_center[length(rng)-m, n] == 1 # hide
                    scatter!(p, (xi, yi), markersize=15, markershape=:circle, markercolor=RGB(1.0, 0.498039, 0.054902), alpha=0.6, label=false) # hide
                    scatter!(p, (xi, yi), color=RGB(0.839216, 0.152941, 0.156863), marker=:star, markersize=8, label=false) # hide
                elseif (rng[n] < xi < rng[n+1]) && (rng[m] < yi < rng[m+1]) && count_center[length(rng)-m, n] >= 3 # hide
                    scatter!(p, (xi, yi), markersize=15, markershape=:circle, markercolor=RGB(0.121569, 0.466667, 0.705882), alpha=0.5, label=false) # hide
                    scatter!(p, (xi, yi), color=RGB(0.839216, 0.152941, 0.156863), marker=:star, markersize=8, label=false) # hide
                else # hide
                    scatter!(p, (xi, yi), color=RGB(0.839216, 0.152941, 0.156863), marker=:star, markersize=8, label=false) # hide
                end # hide
            end # hide
        end # hide
    end # hide
    savefig(p, "$filename.svg"); nothing # hide
end # hide
```


The following is only use KMean without KMean++ and BKMean.
```@example 3
plot_result(clusters,"KMean - gmd55", "kmean")  # hide
```
![](kmean.svg)

This implementation utilizes KMeans with KMeans++ initialization. The resulting clusters demonstrate improved quality, reflected in a slightly lower SSE.

```@example 3
plot_result(clusters_pp,"KMean with KMean++ - gmd55", "kmean_pp") # hide
```
![](kmean_pp.svg)

When employing KMeans with KMeans++ and BKMeans, a significant enhancement in clustering performance is evident.

```@example 3
plot_result(clusters_pp_b,"KMean with KMean++ and BKMean - gmd55", "kmean_pp_b") # hide
```
![](kmean_pp_b.svg)
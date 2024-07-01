using KMeansClustering
using Test
using Clustering
using Random
using BenchmarkTools
using Statistics
using Plots
using DataFrames
using CSV


println(pwd())
# make sure that test_plot is existed
output_dir = "test_plot"
isdir(output_dir) || mkpath(output_dir)


# Prepare Dataset
# small one
data_small = [rand(2) for _ in 1:2]

# large one
data_large = [rand(2) for _ in 1:500]

# A array which can accept Float64 and missing data
missing_process = Array{Union{Float64, Missing}}(undef, 100, 2)

# fill in randomized data
for i in 1:100
    for j in 1:2
        missing_process[i, j] = rand()
    end
end

missing_process[1, :] .= missing

# deal with missing data
col_means = [mean(skipmissing(missing_process[:, j])) for j in 1:size(missing_process, 2)]
for j in eachindex(col_means)
    missing_process[ismissing.(missing_process[:, j]), j] .= col_means[j]
end

#type convert
data_missing = [vec(convert(Array{Float64}, missing_process[i, :])) for i in 1:size(missing_process, 1)]


# with extreme datya
x = rand(100, 2)  # 2-dims

for i in 1:5 
    x[i, :] .= 1000 + i
end

#Vector{Vector{Float64}}
data_outlier = [vec(x[i, :]) for i in 1:size(x, 1)]

# Vector{Vector{Float64}}
function convert_to_vector(data::DataFrame)
    vector_data = [Vector{Float64}(row) for row in eachrow(data)]
    return vector_data
end

#Load Iris
function load_iris_data()
    iris_data = load_iris()
    iris_df = DataFrame(iris_data)
    print(iris_df)
    y_iris,x_iris = unpack(iris_df, ==(:target); rng=123)
    vector = x_iris[:,1:2]
    println("iris_vectors:",vector)

    iris_vectors = convert_to_vector(vector)

    iris_labels = y_iris

    iris_label_map = Dict(unique(iris_labels) .=> 1:length(unique(iris_labels)))
    iris_int_labels = [iris_label_map[label] for label in iris_labels]
    println("length of iris_vectors: ",length(iris_vectors))
    println("length of iris_int_labels: ",length(iris_int_labels))

    println("iris_vectors, iris_int_labels",iris_vectors, iris_int_labels)
    return iris_vectors, iris_int_labels
end


function load_wine_data()
    # wine
    # there are 13 feature: Alcohol,Malic.acid,Ash,Acl,Mg,Phenols,Flavanoids,Nonflavanoid.phenols,Proanth,Color.int,Hue,OD,Proline
    # url = "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv"
    data_path = "./test/wine.csv"

    wine_df = CSV.read(data_path, DataFrame)
    y_wine, X_wine = unpack(wine_df, ==(:Wine); rng=123);
    vector = X_wine[:,1:2]
    println("wine_vectors:",vector)

    wine_vectors = convert_to_vector(vector)

    wine_labels = y_wine

    wine_label_map = Dict(unique(wine_labels) .=> 1:length(unique(wine_labels)))
    wine_int_labels = [wine_label_map[label] for label in wine_labels]
    println("length of wine_vectors: ",length(wine_vectors))
    println("length of wine_int_labels: ",length(wine_int_labels))

    return wine_vectors, wine_int_labels
end


function extract_assignments(data, clusters)
    println("length(data):",length(data))
    assignments = fill(0, length(data))  
    matched = fill(false, length(data))

    for (i, cluster) in enumerate(values(clusters))
        println("Processing cluster $i with points: ", length(cluster))
        for point in cluster
            # match up by index
            index = findfirst(eachindex(data)) do j
                data[j] == point && !matched[j]
            end            
            if isnothing(index)
                error("Failed to find index for point: ", point)
            end
            if assignments[index] != 0
                println("Warning: Overwriting assignment at index $index")
            end
            matched[index] = true  # has matched
            assignments[index] = i
        end
    end
    return assignments
end



function adjusted_rand_index(labels_true, labels_pred)
    n = length(labels_true)
    unique_labels_true = unique(labels_true)
    unique_labels_pred = unique(labels_pred)

    contingency_matrix = zeros(Int, length(unique_labels_true), length(unique_labels_pred))

    label_true_map = Dict(label => i for (i, label) in enumerate(unique_labels_true))
    label_pred_map = Dict(label => i for (i, label) in enumerate(unique_labels_pred))

    for i in 1:n
        contingency_matrix[label_true_map[labels_true[i]], label_pred_map[labels_pred[i]]] += 1
    end

    println("Contingency Matrix: ")
    println(contingency_matrix)

    comb_matrix = comb.(contingency_matrix)
    sum_comb_c = sum(comb_matrix)
    sum_comb_k = sum(comb.(sum(contingency_matrix, dims=2)))
    sum_comb_j = sum(comb.(sum(contingency_matrix, dims=1)))

    n_comb = comb(n, 2)
    index = sum_comb_c - (sum_comb_k * sum_comb_j / n_comb)
    expected_index = (sum_comb_k * sum_comb_j) / n_comb
    max_index = 0.5 * (sum_comb_k + sum_comb_j)

    println("sum_comb_c: $sum_comb_c")
    println("sum_comb_k: $sum_comb_k")
    println("sum_comb_j: $sum_comb_j")
    println("index: $index")
    println("expected_index: $expected_index")
    println("max_index: $max_index")

    return (index - expected_index) / (max_index - expected_index)
end



function comb(n::Int, k::Int=2)::BigInt
    if n < k
        return BigInt(0)
    end
    return factorial(BigInt(n)) // (factorial(BigInt(k)) * factorial(BigInt(n - k)))
end



# Function to plot clustering results and centroids
function plot_clusters(centroids,title, filename)
    p = plot(title=title, legend=:topright)
    color_palette = palette(:tab10)

    for (i, (centers, members)) in enumerate(centroids)

        mem_x = [members[i][1] for i in 1:length(members)]
        mem_y = [members[i][2] for i in 1:length(members)]
    
        # Plot cluster points
        scatter!(p, mem_x, mem_y, label="Cluster $i", color=color_palette[i])
    
        # Plot cluster center
        scatter!(p, [centers[1]], [centers[2]], color=color_palette[i], marker=:star, markersize=10, label="Center $i")
    
    end

    savefig(filename)
end






@testset "Iris Dataset" begin
    try
        iris_vectors, iris_labels = load_iris_data()
        println("Loaded Iris Data")
        
        k = length(unique(iris_labels))  
        result = KMeans(iris_vectors, k)

        println("KMeans Clustering Completed")
        
        assignments = extract_assignments(iris_vectors, result)
        println("Assignments for iris dataset: ", assignments)
        
        @test length(assignments) == length(iris_labels)
        
        plot_clusters(result, "Clusters for iris Dataset", joinpath(output_dir, "clusters_iris.png"))
        println("Cluster plot saved as clusters_iris.png")
        
        print(iris_labels, assignments)
        ari = adjusted_rand_index(iris_labels, assignments) 

        println("ARI for iris dataset: ", ari) 
        
        @test -1<= ari <=1   
    catch e
        @test false
        println("Error during testing iris dataset: ", e)
    end
end

@testset "wine Dataset" begin
    try
        wine_vectors, wine_labels = load_wine_data()
        println("Loaded wine Data")
        
        k = length(unique(wine_labels))  
        result = KMeans(wine_vectors, k)
        println("KMeans Clustering Completed")
        
        assignments = extract_assignments(wine_vectors, result)
        println("Assignments for wine dataset: ", assignments)
        
        @test length(assignments) == length(wine_labels)
        
        plot_clusters(result, "Clusters for wine Dataset with centroids ", joinpath(output_dir, "clusters_wine.png"))
        println("Cluster plot saved as clusters_wine.png")
        
        print(wine_labels, assignments)
        ari = adjusted_rand_index(wine_labels, assignments) 

        println("ARI for wine dataset: ", ari) 
        
        @test -1<= ari <=1  
    catch e
        @test false
        println("Error during testing wine dataset: ", e)
    end
end


@testset "KMeansClustering.jl" begin
    try
        result_small = KMeans(data_small, 2)
        assignments_result_small = extract_assignments(data_small, result_small)
        println("Assignments for small dataset: ", assignments_result_small)
        @test length(assignments_result_small) == size(data_small, 1)
        # Plot and save image
        plot_clusters(result_small, "Clusters for Small Dataset", joinpath(output_dir, "clusters_small.png"))
    catch e
        @test false 
        println("Error during testing small dataset: ", e)
    end


    try
        result_large = KMeans(data_large, 5)
        assignments_result_large = extract_assignments(data_large, result_large)
        println("Assignments for large dataset: ", assignments_result_large)
        bench_large = @benchmark KMeans(data_large, 5)
        display(bench_large)
        @test length(assignments_result_large) == size(data_large, 1)
        plot_clusters(result_large, "Clusters for Large Dataset", joinpath(output_dir, "clusters_large.png"))
    catch e
        @test false
        println("Error during testing large dataset: ", e)
    end

    try
        result_missing = KMeans(data_missing, 3)
        assignments_result_missing = extract_assignments(data_missing, result_missing)
        println("Assignments for dataset with missing values: ", assignments_result_missing)
        @test length(assignments_result_missing) == size(data_missing, 1)
        plot_clusters(result_missing, "Clusters for Dataset with Missing Values", joinpath(output_dir, "clusters_missing.png"))
    catch e
        @test false
        println("Error during testing dataset with missing values: ", e)
    end

    try
        result_outlier = KMeans(data_outlier, 3)
        assignments_result_outlier = extract_assignments(data_outlier, result_outlier)
        println("Assignments for dataset with outliers: ", assignments_result_outlier)
        @test length(assignments_result_outlier) == size(data_outlier, 1)
        plot_clusters(result_outlier, "Clusters for Dataset with Outliers", joinpath(output_dir, "clusters_outlier.png"))
    catch e
        @test false
        println("Error during testing dataset with outliers: ", e)
    end
end

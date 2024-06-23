using KMeansClustering
using Test
using Clustering
using Random
using BenchmarkTools
using Statistics
using Plots
using DataFrames



println(pwd())
# make sure that test_plot is existed
output_dir = "test_plot"
isdir(output_dir) || mkpath(output_dir)



# Prepare Dataset
# small one
# data_small = rand(2, 2)
# Output: [[0.343, 0.879], [0.576, 0.123]]
data_small = [rand(2) for _ in 1:2]

# large one
# data_large = rand(10000, 100) 
data_large = [rand(20) for _ in 1:1000]
#Vector{Vector{Float64}}
# typeof(data_large)

# with missing data


# A array which can accept Float64 and missing data
missing_process = Array{Union{Float64, Missing}}(undef, 100, 10)

# fill in randomized data
for i in 1:100
    for j in 1:10
        missing_process[i, j] = rand()
    end
end

# introduce some missing data
missing_process[1, :] .= missing

# deal with missing data
col_means = [mean(skipmissing(missing_process[:, j])) for j in 1:size(missing_process, 2)]
for j in eachindex(col_means)
    missing_process[ismissing.(missing_process[:, j]), j] .= col_means[j]
end

#type convert
data_missing = [vec(convert(Array{Float64}, missing_process[i, :])) for i in 1:size(missing_process, 1)]


typeof(data_missing)
# result = KMeans(data_missing,3)



# with extreme datya
# data_outlier = rand(100, 10)
x = rand(100, 10)  # 2-dims

for i in 1:10  #Diagonal pattern extremes
    x[i, :] .= 1000 + i
end

#Vector{Vector{Float64}}
data_outlier = [vec(x[i, :]) for i in 1:size(x, 1)]


#Functions for extracting cluster assignment results
function extract_assignments(data, clusters)
    assignments = Vector{Int}(undef, length(data))
    for (i, cluster) in enumerate(values(clusters))
        for point in cluster
            index = findfirst(x -> x == point, data)
            assignments[index] = i
        end
    end
    return assignments
end

# ARI No ARI function in Clustering
function adjusted_rand_index(labels_true, labels_pred)
    n = length(labels_true)
    contingency_matrix = zeros(Int, maximum(labels_true), maximum(labels_pred))
    for i in 1:n
        contingency_matrix[labels_true[i], labels_pred[i]] += 1
    end

    sum_comb_c = sum(comb.(contingency_matrix))
    sum_comb_k = sum(comb.(sum(contingency_matrix, dims=2)))
    sum_comb_j = sum(comb.(sum(contingency_matrix, dims=1)))

    n_comb = comb(n, 2)
    index = sum_comb_c - (sum_comb_k * sum_comb_j / n_comb)
    expected_index = (sum_comb_k * sum_comb_j) / n_comb
    max_index = 0.5 * (sum_comb_k + sum_comb_j)
    
    return (index - expected_index) / (max_index - expected_index)
end

function comb(n::Int, k::Int=2)::BigInt
    if n < k
        return BigInt(0)
    end
    return factorial(BigInt(n)) // (factorial(BigInt(k)) * factorial(BigInt(n - k)))
end


# Function to plot clustering results
function plot_clusters(data, assignments, title, filename)
    df = DataFrame(hcat(hcat(data...)', assignments), :auto)
    scatter(df[!, 1], df[!, 2], group=df[!, end], legend=false, title=title)
    savefig(filename)
end


# create the content for markdown
function generate_markdown()
    markdown = """
    ### KMeans Clustering Results

    ![Small Dataset](clusters_small.png)
    ![Large Dataset](clusters_large.png)
    ![Dataset with Missing Values](clusters_missing.png)
    ![Dataset with Outliers](clusters_outlier.png)
    """
    open(joinpath(output_dir, "comment.md"), "w") do f
        write(f, markdown)
    end
end





@testset "KMeansClustering.jl" begin
    try
        result_small = KMeans(data_small, 2)
        assignments_result_small = extract_assignments(data_small, result_small)
        println("Assignments for small dataset: ", assignments_result_small)
        @test length(assignments_result_small) == size(data_small, 1)
        # Plot and save image
        plot_clusters(data_small, assignments_result_small, "Clusters for Small Dataset", joinpath(output_dir, "clusters_small.png"))
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
        # Plot and save image (subset)
        subset = data_large[1:100]
        assignments_subset = assignments_result_large[1:100]
        plot_clusters(subset, assignments_subset, "Clusters for Large Dataset(Subset)", joinpath(output_dir, "clusters_large(subset).png"))
    catch e
        @test false
        println("Error during testing large dataset: ", e)
    end

    try
        result_missing = KMeans(data_missing, 3)
        assignments_result_missing = extract_assignments(data_missing, result_missing)
        println("Assignments for dataset with missing values: ", assignments_result_missing)
        @test length(assignments_result_missing) == size(data_missing, 1)
        plot_clusters(data_missing, assignments_result_missing, "Clusters for Dataset with Missing Values", joinpath(output_dir, "clusters_missing.png"))
    catch e
        @test false
        println("Error during testing dataset with missing values: ", e)
    end

    try
        result_outlier = KMeans(data_outlier, 3)
        assignments_result_outlier = extract_assignments(data_outlier, result_outlier)
        println("Assignments for dataset with outliers: ", assignments_result_outlier)
        @test length(assignments_result_outlier) == size(data_outlier, 1)
        plot_clusters(data_outlier, assignments_result_outlier, "Clusters for Dataset with Outliers", joinpath(output_dir, "clusters_outlier.png"))
    catch e
        @test false
        println("Error during testing dataset with outliers: ", e)
    end
end

# generate Markdown file
generate_markdown()



# ARI test(to test the accuracy)
#in order to ignore error message,just print the result of ARI
@testset "KMeansClustering.jl ARI" begin
    # Assumed that we have actual label
    true_labels_small = [1, 2]
    true_labels_large = repeat(1:10, inner=100)  # length match
    true_labels_missing = repeat(1:3, inner=34)[1:100]  
    true_labels_outlier = repeat(1:3, inner=34)[1:100]  

    try
        result_small = KMeans(data_small, 2)
        assignments_result = extract_assignments(data_small, result_small)
        @test length(assignments_result) == length(true_labels_small)
        ari_small = adjusted_rand_index(true_labels_small, assignments_result)
        println("ARI for small dataset: ", ari_small)
    catch e
        println("Error during ARI calculation for small dataset: ", e)
        @test false
    end

    try
        result_large = KMeans(data_large, 5)
        assignments_result = extract_assignments(data_large, result_large)
        @test length(assignments_result) == length(true_labels_large) 
        ari_large = adjusted_rand_index(true_labels_large, assignments_result)
        println("ARI for large dataset: ", ari_large)
    catch e
        println("Error during ARI calculation for large dataset: ", e)
        @test false
    end

    try
        result_missing = KMeans(data_missing, 3)
        assignments_result = extract_assignments(data_missing, result_missing)
        @test length(assignments_result) == length(true_labels_missing)  
        ari_missing = adjusted_rand_index(true_labels_missing, assignments_result)
        println("ARI for dataset with missing values: ", ari_missing)
    catch e
        println("Error during ARI calculation for dataset with missing values: ", e)
        @test false
    end

    try
        result_outlier = KMeans(data_outlier, 3)
        assignments_result = extract_assignments(data_outlier, result_outlier)
        @test length(assignments_result) == length(true_labels_outlier) 
        ari_outlier = adjusted_rand_index(true_labels_outlier, assignments_result)
        println("ARI for dataset with outliers: ", ari_outlier)
    catch e
        println("Error during ARI calculation for dataset with outliers: ", e)
        @test false
    end
end
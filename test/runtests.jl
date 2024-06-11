# using KMeansClustering
using Clustering
using Random
using BenchmarkTools
using Test
using Statistics
using Plots
using DataFrames

# make sure that test_plot is existed
output_dir = "test/test_plot"
# isdir(output_dir) || mkdir(output_dir)


# Prepare Dataset
# small one
data_small = rand(2, 2)

# large one
data_large = rand(10000, 100) 

# with missing data


# A array which can accept Float64 and missing data
data_missing = Array{Union{Float64, Missing}}(undef, 100, 10)

# fill in randomized data
for i in 1:100
    for j in 1:10
        data_missing[i, j] = rand()
    end
end

# introduce some missing data
data_missing[1, :] .= missing

# deal with missing data
col_means = [mean(skipmissing(data_missing[:, j])) for j in 1:size(data_missing, 2)]
for j in eachindex(col_means)
    data_missing[ismissing.(data_missing[:, j]), j] .= col_means[j]
end

#All Float64，No missing
data_missing = convert(Array{Float64}, data_missing)

# for j in 1:size(data_missing, 2)
#     if any(ismissing.(data_missing[:, j]))
#         data_missing[ismissing.(data_missing[:, j]), j] = mean(skipmissing(data_missing[:, j]))
#     end
# end



# with extreme datya
data_outlier = rand(100, 10)
data_outlier[1, :] .= 1000 


# Function to plot clustering results
function plot_clusters(data, assignments, title, filename)
    df = DataFrame(hcat(data', assignments), :auto)
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
        result_small = kmeans(data_small, 2)
        assignments_result_small = assignments(result_small)
        println("Assignments for small dataset: ", assignments_result_small)
        @test length(assignments_result_small) == size(data_small, 2)
        # Plot and save image
        plot_clusters(data_small, assignments_result_small, "Clusters for Small Dataset", joinpath(output_dir, "clusters_small.png"))
    catch e
        @test false 
        println("Error during testing small dataset: ", e)
    end


    try
        result_large = kmeans(data_large, 10)
        assignments_result_large = assignments(result_large)
        println("Assignments for large dataset: ", assignments_result_large)
        bench_large = @benchmark kmeans(data_large, 10)
        display(bench_large)
        @test length(assignments_result_large) == size(data_large, 2)
        # Plot and save image (subset)
        subset = data_large[:, 1:2]
        assignments_subset = assignments_result_large[1:2]
        plot_clusters(subset, assignments_subset, "Clusters for Large Dataset(Subset)", joinpath(output_dir, "clusters_large(subset).png"))
    catch e
        @test false
        println("Error during testing large dataset: ", e)
    end

    try
        result_missing = kmeans(data_missing, 3)
        assignments_result_missing = assignments(result_missing)
        println("Assignments for dataset with missing values: ", assignments_result_missing)
        @test length(assignments_result_missing) == size(data_missing, 2)
        plot_clusters(data_missing, assignments_result_missing, "Clusters for Dataset with Missing Values", joinpath(output_dir, "clusters_missing.png"))
    catch e
        @test false
        println("Error during testing dataset with missing values: ", e)
    end

    try
        result_outlier = kmeans(data_outlier, 3)
        assignments_result_outlier = assignments(result_outlier)
        println("Assignments for dataset with outliers: ", assignments_result_outlier)
        @test length(assignments_result_outlier) == size(data_outlier, 2)
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
    true_labels_small = [1, 1]
    true_labels_large = repeat(1:10, inner=10)  # length match
    true_labels_missing = repeat(1:3, inner=4)[1:10]  
    true_labels_outlier = repeat(1:3, inner=4)[1:10]  

    try
        result_small = kmeans(data_small, 2)
        assignments_result = assignments(result_small)
        @test length(assignments_result) == length(true_labels_small)
        ari_small = randindex(true_labels_small, assignments_result)
        println("ARI for small dataset: ", ari_small)
    catch e
        println("Error during ARI calculation for small dataset: ", e)
        @test false
    end

    try
        result_large = kmeans(data_large, 10)
        assignments_result = assignments(result_large)
        @test length(assignments_result) == length(true_labels_large) 
        ari_large = randindex(true_labels_large, assignments_result)
        println("ARI for large dataset: ", ari_large)
    catch e
        println("Error during ARI calculation for large dataset: ", e)
        @test false
    end

    try
        result_missing = kmeans(data_missing, 3)
        assignments_result = assignments(result_missing)
        @test length(assignments_result) == length(true_labels_missing)  
        ari_missing = randindex(true_labels_missing, assignments_result)
        println("ARI for dataset with missing values: ", ari_missing)
    catch e
        println("Error during ARI calculation for dataset with missing values: ", e)
        @test false
    end

    try
        result_outlier = kmeans(data_outlier, 3)
        assignments_result = assignments(result_outlier)
        @test length(assignments_result) == length(true_labels_outlier) 
        ari_outlier = randindex(true_labels_outlier, assignments_result)
        println("ARI for dataset with outliers: ", ari_outlier)
    catch e
        println("Error during ARI calculation for dataset with outliers: ", e)
        @test false
    end
end

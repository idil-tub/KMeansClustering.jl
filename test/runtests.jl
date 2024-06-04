#using KMeansClustering
using Clustering
using Random
using BenchmarkTools
using Test
using Statistics

# Prepare Dataset
# small one
data_small = rand(2, 2)

# large one
data_large = rand(10000, 100)

# with missing data
data_missing = rand(100, 10)
data_missing[1, :] .= NaN

#deal with missing data


#fill_value = mean(skipmissing(data_missing), dims=1)

col_means = []
for j in 1:size(data_missing, 2)
    col_mean = mean(skipmissing(data_missing[:, j]))
    push!(col_means, col_mean)
end

fill_value = col_means

for j in eachindex(fill_value)
    data_missing[ismissing.(data_missing[:, j]), j] = fill_value[j]
end

# with extreme datya
data_outlier = rand(100, 10)
data_outlier[1, :] .= 1000 


@testset "KMeansClustering.jl" begin
    # Write your tests here.
     result_small = kmeans(data_small, 2)
     @test length(assignments(result_small)) == 2
 
     result_large = kmeans(data_large, 10)
     bench_large = @benchmark kmeans(data_large, 10)
     display(bench_large)
 
     result_missing = kmeans(data_missing', 3)
 
     result_outlier = kmeans(data_outlier', 3)
 
     #Assumed we have known the true label

     true_labels_small = [1, 1]
     true_labels_large = vcat(fill(1, 1000), fill(2, 1000), fill(3, 8000))
     true_labels_missing = repeat(1:3, inner=33)
     true_labels_outlier = repeat(1:3, inner=33)
 
     # calculate ARI
     ari_small = randindex(true_labels_small, assignments(result_small))
     ari_large = randindex(true_labels_large, assignments(result_large))
     ari_missing = randindex(true_labels_missing, assignments(result_missing))
     ari_outlier = randindex(true_labels_outlier, assignments(result_outlier))
 
    # 1 means perfect

     @test ari_small > 0.75
     @test ari_large > 0.75
     @test ari_missing > 0.75
     @test ari_outlier > 0.75
end
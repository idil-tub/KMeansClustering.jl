using Test
using Distributions
using LinearAlgebra
using Statistics
using Random

using KMeansClustering

# Helper function to generate random data points
function generate_random_points(n, dims)
    return [rand(dims) for _ in 1:n]
end

# Test the UniformRandomInit
@testset "UniformRandomInit" begin
    samples = generate_random_points(100, 2)
    init = UniformRandomInit{Vector{Float64}}()
    centroids = init(samples, 3)   
    
    @test length(centroids) == 3
    @test all(x -> length(x) == 2, centroids)
end


# Test the buildClusters function
@testset "buildClusters" begin
    samples = [rand(2) for _ in 1:100]
    initial_centroids = [rand(2) for _ in 1:3]
    
    clusters = buildClusters(samples, initial_centroids)
    
    @test length(clusters) == 3
    @test all(x -> eltype(x) == Vector{Float64}, clusters)
    
    # Verify that each sample is assigned to the nearest centroid
    for i in 1:length(samples)
        sample = samples[i]
        distances = [norm(sample - centroid) for centroid in initial_centroids]
        min_index = argmin(distances)
        @test sample in clusters[min_index]
    end
end


# Test the calculateCenter function
@testset "calculateCenter" begin
    samples = [rand(2) for _ in 1:10]
    center = calculateCenter(samples) 

    @test length(center) == 2
    @test typeof(center) == Vector{Float64}

    # Verify center is the mean of samples
    calculated_mean = mean(samples)
    @test isapprox(center, calculated_mean, atol=1e-6)
end

# Test the KMeans function
@testset "KMeans" begin
    samples = [rand(2) for _ in 1:100]
    k = 3
    result = KMeans(samples, k)
    
    @test length(result) == k

    for (center, cluster) in result
        @test typeof(center) == Vector{Float64}
        @test eltype(cluster) == Vector{Float64}
    end
end


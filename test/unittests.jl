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

# Test the UniformRandomInit with hardcoded data
@testset "UniformRandomInit with hardcoded data" begin
    samples = [ [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
                [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0] ]
    init = UniformRandomInit{Vector{Float64}}()
    centroids = init(samples, 3)
    
    @test length(centroids) == 3
    @test all(x -> length(x) == 2, centroids)
end

# Test the UniformRandomInit with randomly generated data
@testset "UniformRandomInit with random data" begin
    samples = generate_random_points(100, 2)
    init = UniformRandomInit{Vector{Float64}}()
    centroids = init(samples, 3)   
    
    @test length(centroids) == 3
    @test all(x -> length(x) == 2, centroids)
end

# Test the buildClusters function with hardcoded data
@testset "buildClusters with hardcoded data" begin
    samples = [ [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
                [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0] ]
    initial_centroids = [ [2.0, 3.0], [5.0, 6.0], [8.0, 9.0] ]
    
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

# Test the buildClusters function with randomly generated data
@testset "buildClusters with random data" begin
    samples = generate_random_points(100, 2)
    initial_centroids = generate_random_points(3, 2)
    
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

# Test the calculateCenter function with hardcoded data
@testset "calculateCenter with hardcoded data" begin
    samples = [ [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0] ]
    center = calculateCenter(samples)

    @test length(center) == 2
    @test typeof(center) == Vector{Float64}

    # Verify center is the mean of samples
    calculated_mean = mean(samples)
    @test isapprox(center, calculated_mean, atol=1e-6)
end

# Test the calculateCenter function with randomly generated data
@testset "calculateCenter with random data" begin
    samples = generate_random_points(10, 2)
    center = calculateCenter(samples) 

    @test length(center) == 2
    @test typeof(center) == Vector{Float64}

    # Verify center is the mean of samples
    calculated_mean = mean(samples)
    @test isapprox(center, calculated_mean, atol=1e-6)
end

# Test the KMeans function with hardcoded data
@testset "KMeans with hardcoded data" begin
    samples = [ [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
                [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0] ]
    k = 3
    result = KMeans(samples, k)
    
    @test length(result) == k

    for (center, cluster) in result
        @test typeof(center) == Vector{Float64}
        @test eltype(cluster) == Vector{Float64}
    end
end

# Test the KMeans function with randomly generated data
@testset "KMeans with random data" begin
    samples = generate_random_points(100, 2)
    k = 3
    result = KMeans(samples, k)
    
    @test length(result) == k

    for (center, cluster) in result
        @test typeof(center) == Vector{Float64}
        @test eltype(cluster) == Vector{Float64}
    end
end

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
end

# Test the calculateCenter function
@testset "calculateCenter" begin
    samples = [rand(2) for _ in 1:10]
    center = calculateCenter(samples)  
    @test length(center) == 2
    @test typeof(center) == Vector{Float64}

    samples2 = [rand(3) for _ in 1:10]    
    center2 = calculateCenter(samples2)   
    @test length(center2) == 3
    @test typeof(center2) == Vector{Float64}
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

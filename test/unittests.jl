using Test
using Distributions
using Statistics
using Random
using KMeansClustering

# Helper function to generate random data points
function generate_random_points(n, dims)
    return [rand(dims) for _ in 1:n]
end

# Test the UniformRandomInit with hardcoded data
@testset "UniformRandomInit with hardcoded data" begin
    samples = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
        [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0]]
    init = UniformRandomInit{Vector{Float64}}()
    norm = EuclideanNorm{Vector{Float64}}()
    centroids = init(samples, 3, norm)

    @test length(centroids) == 3
    @test all(x -> length(x) == 2, centroids)
end

# Test the KMeansPPInit with hardcoded data
@testset "KMeansPPInit with hardcoded data" begin
    samples = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
        [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0]]
    init = KMeansPPInit{Vector{Float64}}()
    norm = EuclideanNorm{Vector{Float64}}()
    centroids = init(samples, 3, norm)

    @test length(centroids) == 3
    @test all(x -> length(x) == 2, centroids)
end



# Test the UniformRandomInit with randomly generated data
@testset "UniformRandomInit with random data" begin
    samples = generate_random_points(100, 2)
    init = UniformRandomInit{Vector{Float64}}()
    norm = EuclideanNorm{Vector{Float64}}()
    centroids = init(samples, 3, norm)

    @test length(centroids) == 3
    @test all(x -> length(x) == 2, centroids)
end

@testset "UniformRandomInit with flat Vector{Float64}" begin
    samples = rand(Float64, 100)
    init = UniformRandomInit{Float64}()
    norm = EuclideanNorm{Float64}()
    centroids = init(samples, 3, norm)

    @test length(centroids) == 3
    @test all(x -> x isa Float64, centroids)
end

# Test the KMeansPPInit with randomly generated data
@testset "KMeansPPInit with random data" begin
    samples = generate_random_points(100, 2)
    init = KMeansPPInit{Vector{Float64}}()
    norm = EuclideanNorm{Vector{Float64}}()
    centroids = init(samples, 3, norm)

    @test length(centroids) == 3
    @test all(x -> length(x) == 2, centroids)
end

@testset "UniformRandomInit with flat Vector{Float64}" begin
    samples = rand(Float64, 100)
    init = UniformRandomInit{Float64}()
    norm = EuclideanNorm{Float64}()
    centroids = init(samples, 3, norm)

    @test length(centroids) == 3
    @test all(x -> x isa Float64, centroids)
end

# Test the buildClusters function with hardcoded data
@testset "buildClusters with hardcoded data" begin
    samples = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
        [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0]]
    initial_centroids = [[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]]
    norm = EuclideanNorm{Vector{Float64}}()

    clusters = KMeansClustering.buildClusters(samples, initial_centroids, norm)

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
    norm = EuclideanNorm{Vector{Float64}}()

    clusters = KMeansClustering.buildClusters(samples, initial_centroids, norm)

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
    samples = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
    norm = EuclideanNorm{Vector{Float64}}()
    centroidCalculator = EuclideanMeanCentroid{Vector{Float64}}()
    center = centroidCalculator(samples, norm)

    @test length(center) == 2
    @test typeof(center) == Vector{Float64}

    # Verify center is the mean of samples
    calculated_mean = mean(samples)
    @test isapprox(center, calculated_mean, atol=1e-6)
end

# Test the calculateCenter function with randomly generated data
@testset "calculateCenter with random data" begin
    samples = generate_random_points(10, 2)
    norm = EuclideanNorm{Vector{Float64}}()
    centroidCalculator = EuclideanMeanCentroid{Vector{Float64}}()
    center = centroidCalculator(samples, norm)

    @test length(center) == 2
    @test typeof(center) == Vector{Float64}

    # Verify center is the mean of samples
    calculated_mean = mean(samples)
    @test isapprox(center, calculated_mean, atol=1e-6)
end

@testset "KMeans" begin
    # Test the KMeans function with hardcoded data
    inits = [UniformRandomInit, KMeansPPInit]
    for i in 1:2
        @testset "KMeans with hardcoded data init: $(i == 1 ? "UniformRandomInit" : "KMeans++")" begin
            samples = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
                [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0]]
            k = 3
            init = i == 1 ? UniformRandomInit{Vector{Float64}}() : KMeansPPInit{Vector{Float64}}()
            result = KMeans(samples, k, init=init)

            @test length(result) == k

            for (center, cluster) in result
                @test typeof(center) == Vector{Float64}
                @test eltype(cluster) == Vector{Float64}
            end
        end

        # Test the KMeans function with randomly generated data
        @testset "KMeans with random data init: $(i == 1 ? "UniformRandomInit" : "KMeans++")" begin
            samples = generate_random_points(100, 2)
            k = 3
            init = i == 1 ? UniformRandomInit{Vector{Float64}}() : KMeansPPInit{Vector{Float64}}()
            result = KMeans(samples, k, init=init)

            @test length(result) == k

            for (center, cluster) in result
                @test typeof(center) == Vector{Float64}
                @test eltype(cluster) == Vector{Float64}
            end
        end
    end
end

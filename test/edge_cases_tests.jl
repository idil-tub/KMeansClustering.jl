using Test

# Unit tests for kmeans function
@testset "KMeans Tests" begin
    # Basic functionality test
    X = [1.0 2.0; 1.5 1.8; 5.0 8.0; 8.0 8.0]
    labels, centroids = kmeans(X, 2)
    @test length(unique(labels)) == 2
    @test size(centroids, 1) == 2

    # Test with more clusters than data points
    labels, centroids = kmeans(X, 5)
    @test length(unique(labels)) == 4
    @test size(centroids, 1) == 5

    # Test convergence criteria
    labels, centroids = kmeans(X, 2, max_iter=1)
    @test length(unique(labels)) <= 2
    @test size(centroids, 1) == 2

    # Edge case: single cluster
    labels, centroids = kmeans(X, 1)
    @test all(labels .== 1)
    @test size(centroids, 1) == 1

    # Edge case: all points are the same
    Y = [1.0 1.0; 1.0 1.0; 1.0 1.0; 1.0 1.0]
    labels, centroids = kmeans(Y, 2)
    @test length(unique(labels)) == 1
    @test size(centroids, 1) == 1
end

# Unit tests for kmeans_pp_init function
@testset "KMeans++ Initialization Tests" begin
    X = [1.0 2.0; 1.5 1.8; 5.0 8.0; 8.0 8.0]

    # Test correct number of centroids
    centroids = kmeans_pp_init(X, 2)
    @test size(centroids, 1) == 2
    @test size(centroids, 2) == size(X, 2)

    # Test initialization with more clusters than data points
    centroids = kmeans_pp_init(X, 5)
    @test size(centroids, 1) == 5

    # Edge case: single cluster
    centroids = kmeans_pp_init(X, 1)
    @test size(centroids, 1) == 1

    # Edge case: all points are the same
    Y = [1.0 1.0; 1.0 1.0; 1.0 1.0; 1.0 1.0]
    centroids = kmeans_pp_init(Y, 2)
    @test size(centroids, 1) == 2
    @test all(centroids .== 1.0)
end

# Unit tests for adjusted_rand_index function
@testset "Adjusted Rand Index Tests" begin
    labels_true = [1, 1, 2, 2]
    labels_pred = [1, 1, 2, 2]
    @test adjusted_rand_index(labels_true, labels_pred) == 1.0

    labels_pred = [1, 1, 1, 1]
    @test adjusted_rand_index(labels_true, labels_pred) == 0.0

    # Test with different cluster sizes
    labels_true = [1, 1, 1, 2, 2, 2, 3, 3]
    labels_pred = [1, 1, 2, 2, 3, 3, 3, 1]
    @test adjusted_rand_index(labels_true, labels_pred) ≈ 0.2424

    # Edge case: single cluster
    labels_true = [1, 1, 1, 1]
    labels_pred = [1, 1, 1, 1]
    @test adjusted_rand_index(labels_true, labels_pred) == 1.0

    # Edge case: completely different clusters
    labels_true = [1, 1, 1, 1]
    labels_pred = [2, 2, 2, 2]
    @test adjusted_rand_index(labels_true, labels_pred) == 1.0

    # Edge case: no matching clusters
    labels_true = [1, 1, 2, 2]
    labels_pred = [2, 2, 1, 1]
    @test adjusted_rand_index(labels_true, labels_pred) == 0.0
end

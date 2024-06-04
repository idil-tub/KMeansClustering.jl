using Test
using Random
using LinearAlgebra
using Statistics

# Basic implementation of k-means++ initialization
function kmeans_pp(X::Matrix{T}, k::Int) where T<:Real
    n, d = size(X)
    centroids = zeros(T, k, d)
    # Initialize the first centroid randomly
    centroids[1, :] = X[rand(1:n), :]
    
    for i in 2:k
        # Calculate the minimum distance of each point to the closest centroid
        min_distances = [minimum([norm(X[j, :] - centroids[l, :])^2 for l in 1:i-1]) for j in 1:n]
        # Select the next centroid probabilistically proportional to the square distance
        probabilities = min_distances / sum(min_distances)
        cumulative_probabilities = cumsum(probabilities)
        rand_value = rand()
        for j in 1:n
            if rand_value < cumulative_probabilities[j]
                centroids[i, :] = X[j, :]
                break
            end
        end
    end
    
    return centroids
end

function kmeans(X::Matrix{T}, k::Int; max_iters::Int=100, tol::T=1e-4) where T<:Real
    println("Start of kmeans function")
    println("X dimensions: ", size(X))
    n, d = size(X)
    centroids = kmeans_pp(X, k)  # Use k-means++ for initialization
    println("Centroids dimensions: ", size(centroids))
    prev_centroids = similar(centroids)
    labels = similar(Vector{Int}, n)

    for iter in 1:max_iters
        println("Iteration: ", iter)
        # Assignment step
        for j in 1:n
            distances = [norm(X[j, :] - centroids[l, :]) for l in 1:k]
            labels[j] = argmin(distances)
        end

        # Ensure each cluster has at least one point
        for l in 1:k
            if sum(labels .== l) == 0
                labels[rand(1:n)] = l
            end
        end

        # Update step
        for l in 1:k
            cluster_points = X[labels .== l, :]
            if size(cluster_points, 1) > 0
                centroids[l, :] .= vec(mean(cluster_points, dims=1))
            end
        end

        # Check convergence
        if norm(centroids - prev_centroids) < tol
            break
        end

        copyto!(prev_centroids, centroids)
    end

    println("End of kmeans function")
    return labels, centroids
end





# Define a function to test KMeans
function test_kmeans()
    println("Start of test_kmeans function")
    # Generate some random data for testing
    Random.seed!(1234)
    data = [rand(Float64, 2) for _ in 1:100]  # Generate Float64 data
    println("Generated data")

    # Check dimensionality of data vectors
    dimensions = unique(length.(data))
    println("Dimensions: ", dimensions)
    @test length(dimensions) == 1  # Ensure all data vectors have the same length

    # Convert vector of vectors to matrix
    data_matrix = hcat(data...)
    println("Converted to data matrix")

    # Define the number of clusters
    k = 3

    # Test KMeans clustering
    labels, centroids = kmeans(data_matrix, k)
    println("Performed KMeans clustering")

    # Check that the number of clusters is correct
    @test size(centroids, 1) == k

    # Check that each centroid has the same dimensionality as the data points
    @test size(centroids, 2) == size(data_matrix, 2)

    # Check that all labels are valid
    @test all(x -> 1 <= x <= k, labels)

    println("End of test_kmeans function")
end

# Run the test
@testset "KMeans Tests" begin
    test_kmeans()
end

# Print the test results
test_kmeans() 
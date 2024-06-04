function test_kmeans()
    # Generate some sample data for testing
    X = [randn(100, 2) .+ [5, 5]; randn(100, 2) .+ [-5, -5]]
    
    # Initialize KMeans with 2 clusters
    kmeans = KMeans(n_clusters=2)
    
    # Fit KMeans to the data
    kmeans.fit(X)
    
    # Check if the cluster centers have been computed
    @test length(kmeans.cluster_centers_) == 2
    
    # Check if labels have been assigned to each point
    @test length(kmeans.labels_) == size(X, 1)
    
    # Check if labels are integers
    @test typeof(kmeans.labels_[1]) == Int
    
    # Check if labels are within the expected range
    @test all(1 .<= kmeans.labels_ .<= 2)
    
    # Check if all labels are distinct
    @test length(unique(kmeans.labels_)) == 2
    
    # Check if cluster centers are within the range of the data
    @test all(kmeans.cluster_centers_ .>= minimum(X) .&& kmeans.cluster_centers_ .<= maximum(X))
end
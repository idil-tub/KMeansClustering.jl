### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 6a575e14-3f91-11ef-2281-0d1eed2652a6
begin
	using Revise, Pkg
	Pkg.develop(path=pwd())
	Pkg.add("Distributions")
	Pkg.add("Plots")
	using Random
	using Distributions
	using Plots
	using KMeansClustering
	using LinearAlgebra
end

# ╔═╡ cd5338d4-cf92-456e-8969-c2c174c6d91a
begin
	import LinearAlgebra.norm_sqr
	Σ = [1.0 0.0; 0.0 8.0]  # Covariance matrix
	
	all_samples = Vector{Vector{Float64}}()
	for i in 1:3
		for j in 1:5
			μ = [(j-1)*10.0, (i-1)*20.0]
			dist = MvNormal(μ, Σ)
        	samples = [rand(dist) for _ in 1:1000]
        	append!(all_samples, samples)
		end
	end
	# @info samples
	scatter(map(s -> s[1], all_samples), map(s -> s[2], all_samples), aspect_ratio=:equal, marker=(:circle, 1, 1.0),)
	kmeans = KMeans(all_samples, 30, algorithm=BkMeans{Vector{Float64}}(8, 2.0))
	function errors(c)
		centers = map(c -> c.first, c)
		total_err = 0.0
		for sample in all_samples
			min_dist = typemax(Float64)
			for center in centers
				min_dist = min(min_dist, norm_sqr(center-sample))
			end
			total_err += min_dist
		end
		return total_err
	end
	centers = map(c -> c.first, kmeans)
	@info errors(kmeans)
	calc_cluster_error = function(center, cluster)
        return sum([norm_sqr(center-el) for el in cluster])
    end
	function buildClusters(xs, init)
	    num_clusters = length(init)
	    clusters = [[] for _ in 1:num_clusters]
	    for x in xs
	        min_dst = typemax(Float64)
	        min_index = 0
	        for i in 1:num_clusters
	            d = norm_sqr(x - init[i])
	            if d < min_dst
	                min_dst = d
	                min_index = i
	            end
	        end
	        push!(clusters[min_index], x)
	    end
	    return clusters
	end
	calc_err = function(centers)
        num_clusters = length(centers)
        err = 0.0
        for sample in all_samples
            min_dst = typemax(Float64)
            for i in 1:num_clusters
                d = norm_sqr(sample - centers[i])
                if d < min_dst
                    min_dst = d
                end
            end
            err += min_dst
        end
        return err
    end
	calc_utilities = function(centers)
        total_error = calc_err(centers)
        utilities = []
        for i in 1:length(centers)
            push!(utilities, calc_err([x for (j, x) in enumerate(centers) if i != j]) - total_error)
        end
        return utilities
        
    end
	clusters = buildClusters(all_samples, centers)
	utilities = calc_utilities(centers)
	for (center, cluster, util) in zip(centers, clusters, utilities)
		@info center, calc_cluster_error(center, cluster), util
	end
	scatter!(map(s -> s[1], centers), map(s -> s[2], centers), marker=(:star, 8, 1.0))

	# kmeans = KMeans(all_samples, 30)
	# centers = map(c -> c.first, kmeans)
	# @info errors(kmeans)
	# scatter!(map(s -> s[1], centers), map(s -> s[2], centers), marker=(:star, 8, 1.0))
	
end

# ╔═╡ Cell order:
# ╠═6a575e14-3f91-11ef-2281-0d1eed2652a6
# ╠═cd5338d4-cf92-456e-8969-c2c174c6d91a

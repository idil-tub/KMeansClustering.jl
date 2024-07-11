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
end

# ╔═╡ cd5338d4-cf92-456e-8969-c2c174c6d91a
begin
	Σ = [1.0 0.0; 0.0 8.0]  # Covariance matrix
	
	all_samples = Vector{Vector{Float64}}()
	for i in 1:5
		for j in 1:10
			μ = [(j-1)*10.0, (i-1)*20.0]
			dist = MvNormal(μ, Σ)
        	samples = [rand(dist) for _ in 1:100]
        	append!(all_samples, samples)
		end
	end
	# @info samples
	scatter(map(s -> s[1], all_samples), map(s -> s[2], all_samples), aspect_ratio=:equal, marker=(:circle, 1, 1.0),)

	kmeans = KMeans(all_samples, 100, algorithm=BkMeans{Vector{Float64}}())
	
end

# ╔═╡ Cell order:
# ╠═6a575e14-3f91-11ef-2281-0d1eed2652a6
# ╠═cd5338d4-cf92-456e-8969-c2c174c6d91a

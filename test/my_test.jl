using KMeansClustering
using Random
using Distributions
import LinearAlgebra.norm_sqr

Σ = [1.0 0.0; 0.0 8.0]
samples = Vector{Vector{Float64}}()
for i in 1:10
    for j in 1:5
        μ = [(j-1)*10.0, (i-1)*20.0]
        dist = MvNormal(μ, Σ)
        append!(samples, [rand(dist) for _ in 1:100])
    end
end

function errors(c)
    centers = map(c -> c.first, c)
    total_err = 0.0
    for sample in samples
        min_dist = typemax(Float64)
        for center in centers
            min_dist = min(min_dist, norm_sqr(center-sample))
        end
        total_err += min_dist
    end
    return total_err
end

min_m = 0
min_eps = 0.0
min_err = typemax(Float64)
for m in 0:50
    eps = 0.01
        bkmeans= BkMeans{Vector{Float64}}(m, eps)
        res = KMeans(samples, 100, algorithm=bkmeans)
        err = errors(res)
        @info err
        if err < min_err
            global min_err = err
            global min_m = m
            global min_eps = eps
        end
end
@info "m: ", min_m, "eps: ", min_eps


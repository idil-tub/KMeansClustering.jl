using MLJ
using DataFrames
using Random
using LinearAlgebra

# use iris dataset first two feature for testing
include("../test/dataset.jl")
y, X = iris_dataset()
X = Matrix(X[:, 1:2])

n_centers = 3
n_samples = size(X, 1)

centers = zeros(n_centers, size(X, 2))
indices = fill(-1, n_centers)
potentialDensityFactor = 1
n_local_trials = 2 + Int(floor(log(length(centers))))

# select the first center randomly
center_id = rand(1:n_samples)
centers[1, :] .= X[center_id, :]
indices[1, 1] = center_id

# closet dist
Euclidean(x, y) = sqrt(sum((x - y).^2))
global closest_dist_sq = []
for i in 1:size(X, 1)
    push!(closest_dist_sq, Euclidean(X[i, :], centers[1, :]))
end

global individualPots = closest_dist_sq * potentialDensityFactor
global current_pot = sum(individualPots)


# Pick the remaining n_centers-1 points
for i in 2:n_centers
    rand_vals = rand(n_local_trials) * current_pot
    candidate_ids = searchsortedlast.(Ref(cumsum(individualPots)), rand_vals)
    clamp!(candidate_ids, 1, length(individualPots))

    distance_to_candidates = []
    candidates_pot = []
    for k in 1:size(candidate_ids, 1)
        distance_to_candidates_part = []
        for j in 1:size(X, 1)
            push!(distance_to_candidates_part, Euclidean(X[j, :], X[candidate_ids[k], :]))   
        end
        distance_to_candidates_part = min.(distance_to_candidates_part, closest_dist_sq) 
        distance_to_candidates = push!(distance_to_candidates, distance_to_candidates_part)
        individualCandidatePots = distance_to_candidates_part * potentialDensityFactor
        candidates_pot = push!(candidates_pot, sum(individualCandidatePots))
    end 


    best_candidate = argmin(candidates_pot)
    global current_pot = candidates_pot[best_candidate]
    global closest_dist_sq = distance_to_candidates[best_candidate]
    global individualPots = closest_dist_sq * potentialDensityFactor
    best_candidate = candidate_ids[best_candidate]

    centers[i, :] .= X[best_candidate, :]
end
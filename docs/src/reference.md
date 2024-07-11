# Public Documentation

Documentation for `KMeansClustering.jl`'s public interface.

## Contents

```@contents
Pages = ["reference.md"]
```

## KMeansClustering

```@docs
KMeansClustering
```

### Functions
```@autodocs
Modules = [KMeansClustering]
Filter = t -> typeof(t) <: Function
```

## Init
```@docs
KMeansClustering.Init
```

### Concrete Implementation
```@autodocs
Modules = [KMeansClustering.Init]
Private = false
Filter = t -> nameof(t) !== :ClusterInit && !(typeof(t) <: Module)
```

### Abstract Initializer
```@autodocs
Modules = [KMeansClustering.Init]
Private = false
Filter = t -> nameof(t) === :ClusterInit 
```

## Norm
```@docs
KMeansClustering.Norm
```

### Concrete Implementation
```@autodocs
Modules = [KMeansClustering.Norm]
Private = false
Filter = t -> nameof(t) !== :NormSqr && !(typeof(t) <: Module)
```

### Abstract Norm
```@autodocs
Modules = [KMeansClustering.Norm]
Private = false
Filter = t -> nameof(t) === :NormSqr 
```

## Centroid
```@docs
KMeansClustering.Centroid
```

### Concrete Implementation
```@autodocs
Modules = [KMeansClustering.Centroid]
Private = false
Filter = t -> nameof(t) !== :CentroidCalculator && !(typeof(t) <: Module)
```

### Abstract Centroid
```@autodocs
Modules = [KMeansClustering.Centroid]
Private = false
Filter = t -> nameof(t) === :CentroidCalculator 
```

## Algorithms
```@docs
KMeansClustering.KMeansAlgorithms
```

### Concrete Implementation
```@autodocs
Modules = [KMeansClustering.KMeansAlgorithms]
Private = false
Filter = t -> nameof(t) !== :KMeansAlgorithm && !(typeof(t) <: Module)
```

### Abstract KMeansAlgorithms
```@autodocs
Modules = [KMeansClustering.KMeansAlgorithms]
Private = false
Filter = t -> nameof(t) === :KMeansAlgorithm 
```
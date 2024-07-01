using KMeansClustering
using Documenter

using Pkg
Pkg.add(["MLJ", "DataFrames", "Plots"])

# Precompile added packages
using MLJ
using DataFrames
using Plots


DocMeta.setdocmeta!(KMeansClustering, :DocTestSetup, :(using KMeansClustering); recursive=true)

makedocs(;
    modules=[KMeansClustering],
    authors="Idil Bilge Can, Yifan Zheng, Lu-Wen Wang, Tristan Kobusch",
    sitename="KMeansClustering.jl",
    format=Documenter.HTML(;
        canonical="https://idil-tub.github.io/KMeansClustering.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
    ],
)

deploydocs(;
    repo="github.com/idil-tub/KMeansClustering.jl",
    devbranch="main",
)

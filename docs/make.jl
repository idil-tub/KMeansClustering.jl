using KMeansClustering
using Documenter

# Precompile added packages
using MLJ
using DataFrames
using Plots
using TSne
using Random
using HTTP
using CSV

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
        "Installation" => "installation.md",
        "Public API" => "reference.md"
    ],
    checkdocs=:exports
)

deploydocs(;
    repo="github.com/idil-tub/KMeansClustering.jl",
    devbranch="main",
)


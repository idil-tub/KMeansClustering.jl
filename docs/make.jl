using KMeansClustering
using Documenter

DocMeta.setdocmeta!(KMeansClustering, :DocTestSetup, :(using KMeansClustering); recursive=true)

makedocs(;
    modules=[KMeansClustering],
    authors=["Idil Bilge Can", "Yifan Zheng", "Lu-Wen Wang", "Tristan Kobusch"],
    sitename="KMeansClustering.jl",
    format=Documenter.HTML(;
        canonical="https://idil-tub.github.io/KMeansClustering.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/idil-tub/KMeansClustering.jl",
    devbranch="main",
)

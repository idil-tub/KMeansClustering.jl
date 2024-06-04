using MLJ
using DataFrames
using HTTP
using CSV

using Plots

# iris
models() 
data = load_iris()
iris = DataFrame(data)

# wine
url = "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv"
data_path = "wine.csv"
HTTP.download(url, data_path)

wine_df = CSV.read(data_path, DataFrame)

# make_blobs
n_samples = 100
n_features = 2
n_centers = 3
X_blobs, y_blobs = make_blobs(n_samples, n_features; centers=n_centers, rng=42)

scatter(X_blobs[1], X_blobs[2])

# make_moons
X_moons, y_moons = make_moons(100; noise=0.05)

scatter(X_moons[1], X_moons[2])

# make_circle
X_circles, y_circles = make_circles(100; noise=0.05, factor=0.5)

scatter(X_circles[1], X_circles[2])
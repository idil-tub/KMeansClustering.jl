using MLJ
using DataFrames
using HTTP
using CSV
using Plots

# make directory for save images
mkpath(".\\src\\image")
mkpath(".\\src\\image\\iris") # dir for saving iris images
mkpath(".\\src\\image\\wine") # dir for saving wine images


# create image from dataset
function create_image(data, data_feature_size, folder_name)
    plot_list = []
    font_size = data_feature_size * 4
    for i in 1:data_feature_size
        for j in 1:data_feature_size
            if i == j
                plot_name = names(data)[i]
                p = plot()
                annotate!(p, 0.5, 0.5, text(plot_name, :center, font_size))
                push!(plot_list, p)
            elseif j < i
                plot_name = string(names(data)[i], "_", names(data)[j])
                p = plot(xlabel=names(data)[i], ylabel=names(data)[j])
                scatter!(p, data[:, i], data[:, j])
                push!(plot_list, p)    
            else 
                plot_name = string(names(data)[i], "_", names(data)[j])
                p = plot(xlabel=names(data)[i], ylabel=names(data)[j])
                scatter!(p, data[:, i], data[:, j])
                savefig(p, ".\\src\\image\\$folder_name\\$plot_name.png")
                push!(plot_list, p) 
            end
        end
    end
    img_length = data_feature_size * 800
    img_width = data_feature_size * 600
    combined_wine_plot = plot(plot_list..., layout=(data_feature_size, data_feature_size), size=(img_length, img_width))
    savefig(combined_wine_plot, ".\\src\\image\\$folder_name\\combined_$folder_name.png")
end

# iris
# there are 4 feature: sepal_length, sepal_width, petal_length, petal_width
# how to use data: ex -> X.sepal_length
models() 
data = load_iris()
iris = DataFrame(data)
iris_y, iris_X = unpack(iris, ==(:target); rng=123);

create_image(iris_X, size(iris_X, 2), "iris")


# wine
# there are 13 feature: Alcohol,Malic.acid,Ash,Acl,Mg,Phenols,Flavanoids,Nonflavanoid.phenols,Proanth,Color.int,Hue,OD,Proline
# how to use data: ex -> wine_df.Alcohol or wine_df[:, 2]
url = "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv"
data_path = ".\\src\\wine.csv"
HTTP.download(url, data_path)

wine_df = CSV.read(data_path, DataFrame)
wine_y, wine_X = unpack(wine_df, ==(:Wine); rng=123);
create_image(wine_X, size(wine_X, 2), "wine")


# make_blobs
# how to use data: X_blobs[1] and X_blobs[2]
n_samples = 100
n_features = 2
n_centers = 3
X_blobs, y_blobs = make_blobs(n_samples, n_features; centers=n_centers, rng=42)

blob = plot(xlabel="X_blobs[1]", ylabel="X_blobs[2]")
scatter!(blob, X_blobs[1], X_blobs[2])
savefig(blob, ".\\src\\image\\blobs.png")

# make_moons
# how to use data: X_moons[1] and X_moons[2]
X_moons, y_moons = make_moons(100; noise=0.05)

moons = plot(xlabel="X_moons[1]", ylabel="X_moons[2]")
scatter!(moons, X_moons[1], X_moons[2])
savefig(moons, ".\\src\\image\\moons.png")


# make_circle
# how to use data: X_circles[1] and X_circles[2]
X_circles, y_circles = make_circles(100; noise=0.05, factor=0.5)

circles = plot(xlabel="X_circles[1]", ylabel="X_circles[2]")
scatter!(circles, X_circles[1], X_circles[2])
savefig(circles, ".\\src\\image\\circle.png")



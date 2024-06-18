using MLJ
using DataFrames
using HTTP
using CSV
using Plots

# make directory for save images
mkpath(".\\test\\image")

# create image from dataset
function createImage(data::DataFrame, folder_name::String, combine::Bool=true)
    mkpath(".\\test\\image\\$folder_name")
    plot_list = []
    data_feature_size = size(data, 2)
    font_size = 60/data_feature_size
    for i in 1:data_feature_size
        for j in 1:data_feature_size
            if i == j
                if combine == true
                # plot title in combine image
                plot_name = names(data)[i]
                p = plot()
                annotate!(p, 0.5, 0.5, text(plot_name, :center, size:font_size))
                push!(plot_list, p)
                end
            elseif j < i
                # avoid save same image, but need to save to the list for create combine image 
                plot_name = string(names(data)[i], "_", names(data)[j])
                p = plot(xlabel=names(data)[i], ylabel=names(data)[j])
                scatter!(p, data[:, i], data[:, j])
                push!(plot_list, p)    
            else 
                # save image
                plot_name = string(names(data)[i], "_", names(data)[j])
                p = plot(xlabel=names(data)[i], ylabel=names(data)[j])
                scatter!(p, data[:, i], data[:, j])
                savefig(p, ".\\test\\image\\$folder_name\\$plot_name.png")
                push!(plot_list, p) 
            end
        end
    end
    if combine == true
        img_length = data_feature_size * 800
        img_width = data_feature_size * 600
        combined_wine_plot = plot(plot_list..., layout=(data_feature_size, data_feature_size), size=(img_length, img_width))
        savefig(combined_wine_plot, ".\\test\\image\\$folder_name\\combined_$folder_name.png")
    end
end

function iris_dataset()
    # iris
    # there are 4 feature: sepal_length, sepal_width, petal_length, petal_width
    models() 
    data = load_iris()
    iris = DataFrame(data)
    y_iris, X_iris = unpack(iris, ==(:target); rng=123);
    return y_iris, X_iris  
end
# createImage(iris_X, "iris")

function wine_dataset()
    # wine
    # there are 13 feature: Alcohol,Malic.acid,Ash,Acl,Mg,Phenols,Flavanoids,Nonflavanoid.phenols,Proanth,Color.int,Hue,OD,Proline
    url = "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv"
    data_path = ".\\test\\wine.csv"
    HTTP.download(url, data_path)

    wine_df = CSV.read(data_path, DataFrame)
    y_wine, X_wine = unpack(wine_df, ==(:Wine); rng=123);
    return y_wine, X_wine
end

function blobs()
    # make_blobs
    n_samples = 100
    n_features = 2
    n_centers = 3
    X_blobs, y_blobs = make_blobs(n_samples, n_features; centers=n_centers, rng=42)
    X_blobs_df = DataFrame(X_blobs)
    return y_blobs, X_blobs_df 
end

function moons()
    # make_moons
    X_moons, y_moons = make_moons(100; noise=0.05)
    X_moons_df = DataFrame(X_moons)
    return y_moons, X_moons_df 
end

function circle()
    # make_circle
    X_circles, y_circles = make_circles(100; noise=0.05, factor=0.5)
    X_circles_df = DataFrame(X_circles)
    return y_circles, X_circles_df 
end

# TSNE
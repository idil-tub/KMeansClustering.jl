using MLJ
using DataFrames
using HTTP
using CSV
using Plots

# make directory for save images
mkpath(".\\src\\image")
mkpath(".\\src\\image\\iris") # dir for saving iris images
mkpath(".\\src\\image\\wine") # dir for saving wine images

# iris
# there are 4 feature: sepal_length, sepal_width, petal_length, petal_width
# how to use data: ex -> X.sepal_length
models() 
data = load_iris()
iris = DataFrame(data)
y, X = unpack(iris, ==(:target); rng=123);

iris_sl_sw = plot(xlabel="sepal_length", ylabel="sepal_width")
scatter!(iris_sl_sw, X.sepal_length, X.sepal_width)
savefig(iris_sl_sw, ".\\src\\image\\iris\\iris_sl_sw.png")

iris_sl_pl = plot(xlabel="sepal_length", ylabel="petal_length")
scatter!(iris_sl_pl, X.sepal_length, X.petal_length)
savefig(iris_sl_pl, ".\\src\\image\\iris\\iris_sl_pl.png")

iris_sl_pw = plot(xlabel="sepal_length", ylabel="petal_width")
scatter!(iris_sl_pw, X.sepal_length, X.petal_width)
savefig(iris_sl_pw, ".\\src\\image\\iris\\iris_sl_pw.png")

iris_sw_pl = plot(xlabel="sepal_width", ylabel="petal_length")
scatter!(iris_sw_pl, X.sepal_width, X.petal_length)
savefig(iris_sw_pl, ".\\src\\image\\iris\\iris_sw_pl.png")

iris_sw_pw = plot(xlabel="sepal_width", ylabel="petal_width")
scatter!(iris_sw_pw, X.sepal_width, X.petal_width)
savefig(iris_sw_pw, ".\\src\\image\\iris\\iris_sw_pw.png")

iris_pl_pw = plot(xlabel="petal_length", ylabel="petal_width")
scatter!(iris_pl_pw, X.petal_length, X.petal_width)
savefig(iris_pl_pw, ".\\src\\image\\iris\\iris_pl_pw.png")

iris_sl = plot()
annotate!(iris_sl, 0.5, 0.5, text("sepal_length", :center, 20))

iris_sw = plot()
annotate!(iris_sw, 0.5, 0.5, text("sepal_width", :center, 20))

iris_pl = plot()
annotate!(iris_pl, 0.5, 0.5, text("petal_length", :center, 20))

iris_pw = plot()
annotate!(iris_pw, 0.5, 0.5, text("petal_width", :center, 20))

combined_iris_plot = plot(iris_sl, iris_sl_sw, iris_sl_pl, iris_sl_pw, 
                          iris_sl_sw, iris_sw, iris_sw_pl, iris_sw_pw,
                          iris_sl_pl, iris_sw_pl, iris_pl, iris_pl_pw,
                          iris_sl_pw, iris_sw_pw, iris_pl_pw, iris_pw,
                          layout=(4,4), 
                          size=(1200,1200))
savefig(combined_iris_plot, ".\\src\\image\\iris\\combined_iris_plot.png")

# wine
# there are 13 feature: Alcohol,Malic.acid,Ash,Acl,Mg,Phenols,Flavanoids,Nonflavanoid.phenols,Proanth,Color.int,Hue,OD,Proline
# how to use data: ex -> wine_df.Alcohol or wine_df[:, 2]
url = "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv"
data_path = ".\\src\\wine.csv"
HTTP.download(url, data_path)

wine_df = CSV.read(data_path, DataFrame)

wine_plot_list = []
for i in 2:14
    for j in 2:14
        if i == j
            plot_name = names(wine_df)[i]
            p = plot()
            annotate!(p, 0.5, 0.5, text(plot_name, :center, 180))
            push!(wine_plot_list, p)
        else
            plot_name = string(names(wine_df)[i], "_", names(wine_df)[j])
            p = plot(xlabel=names(wine_df)[i], ylabel=names(wine_df)[j]) #xlabelfontsize=16, ylabelfontsize=16
            scatter!(p, wine_df[:, i], wine_df[:, j])
            savefig(p, ".\\src\\image\\wine\\$plot_name.png")
            push!(wine_plot_list, p)
        end
    end
end
combined_wine_plot = plot(wine_plot_list..., layout=(13,13), size=(24000,24000))
savefig(combined_wine_plot, ".\\src\\image\\wine\\combined_wine_plot.png")

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
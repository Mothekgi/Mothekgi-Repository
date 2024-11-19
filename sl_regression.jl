using CSV, GLM, Plots, TypedTables

data = CSV.File("winxgd.csv")

X = data.xgd90

Y = round.(Int, data.wins)

t = Table(X = X, Y = Y)

#use Plots
gr(size = (1000, 800))

p_scatter = scatter(X, Y, 
    xlims = (-2, 2.5),
    ylims = (0, 40),
    xlabel = "xgd90",
    ylabel = "wins per season",
    title = "xgd90 - wins",
    legend = false,
    color = :red
)

ols = lm(@formula(Y ~ X), t)

plot!(X, predict(ols), color = :green, linewidth = 2)

#predict wins based on xg difference

newX = Table(X = [1.4])

predict(ols, newX)

using Statistics

 Statistics.cor(X, Y)

 Statistics.cov(X, Y)
############Sasol pricing recurrent neural network

using CSV, DataFrames, Flux, GLMakie

#Loading Sasol daily valuation

data = CSV.read("sasol.csv", DataFrame)

#Close price

price = Float32.(data.Close)

#Vectorise feature and labels
X = price[1:end-1]
Y = price[2:end]

#Transform dimensions
X = [[x] for x ∈ X]

model = Chain(RNN(1 => 32, elu), Dense(32 => 1, identity))

#Model train
epochs = 500
opt = Nesterov()
0 = Flux.params(model) #Model parameters
for epoch ∈ 1:epochs #Traininh loop
    Flux.reset!(model)
    #Loss/Gradient
    ∇ = gradient(0) do 
        model(X[1])
        sum(Flux.Losses.mse.([model(x)[1] for x ∈ X[2:end]], Y[2:end]))
    end
    Flux.update!(opt, 0, ∇)
end

#Coding the price movement animation

df = DataFrame(CSV.File("sasol.csv"))

price = propertynames(df)[5]

fig = Figure(resolution = (1920, 1080))

axis1 = fig[1, 1] = Axis(fig,
    title = "Sasol stock price",
    titlegap = 45, titlesize = 45,
    xgridcolor = :blue, xgridwidth = 3,
    xlabel = "Time(day)", xlabelsize = 45,
    xticklabelsize = 30, xticks = LogTicks(20),
    ygridcolor = :blue, ygridwidth = 3,
    ylabel = "Price", ylabelsize = 45, ytickformat = "{d}",
    yticklabelsize = 30, yticks = LogTicks = (10)
)

frames = 1:(length(df.Date))

colors = [:blue]

#Recording the price movement

record(fig, "Sasol_stock_price.mp4", frames; framerate = 4) do i
    for j in 1:length(Close)
        lines!(axis1, df[1:i, 1], df[1:i, (j + 1)], 
        color = (colors[j], 0.5), linestyle = :solid, linewidth = 2
        )
        scatter!(axis1, df[1:i, 1], df[1:i, (j + 1)], 
        color = colors[j], markersize = 15)
    end
end



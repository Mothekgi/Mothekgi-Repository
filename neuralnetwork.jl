using Flux, MLDatasets, Plots, Statistics, LinearAlgebra

X_train_raw, y_train_raw = MLDatasets.MNIST.traindata(Float32)

X_test_raw, y_test_raw = MLDatasets.MNIST.testdata(Float32)

index = 1

X_train_raw

X_train_raw[:, :, index]

y_train_raw

y_train_raw[index]

X_test_raw

X_test_raw[:, :, index]

y_test_raw

y_test_raw[index]

#parameters

using Flux: crossentropy, onecold, onehotbatch, train!

X_train = Flux.flatten(X_train_raw)
X_test = Flux.flatten(X_test_raw)

#vector concatination

y_train = onehotbatch(y_train_raw, 0:9)

y_test = onehotbatch(y_test_raw, 0:9)

logistic(x) = 1 / (1 + exp(-x))

gr(size = (700, 700))

y_int = 0.0

slope = 1.0

z(x) = y_int .+ slope * x

f(x) = 1 ./ (1 .+ exp.(-z(x)))

X = y_train

Y = y_test


# model engineering

model = Chain(
    Dense(28*28,32,sigmoid),
    Dense(32,10),
    softmax
)

loss_function(x,y) = Flux.crossentropy(model(x),y)

parameters = Flux.params(model)

alpha = 0.011

optimisation = RAdam(alpha)

#model train

loss_history = []

epochs = 700

for epoch in 1:epochs
    train!(loss_function, parameters, [(X_train, y_train)], optimisation)
    train_loss = loss_function(X_train, y_train)
    push!(loss_history, train_loss)
    println("epochs = $epoch : training loss = $train_loss")

end

y_hat_raw = model(X_test)

y_hat = onecold(y_hat_raw) .- 1

y = y_test_raw

mean(y_hat .== y)

#results

results = [y_hat[i] == y[i] for i in 1:length(y)]

index = collect(1:length(y))

results_display = [index y_hat y results]

vscodedisplay(results_display)

#loss plot

gr(size = (700, 700))

plot_alpha_curve = plot(1:epochs, loss_history,
    xlabel = "epochs",
    ylabel = "loss",
    title = "alpha curve",
    legend = false,
    color = :purple,
    linewidth = 3
)

savefig(plot_alpha_curve, "nn_alpha_curve.svg")
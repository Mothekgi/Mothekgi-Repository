using NearestNeighbors, Plots, Random, RDatasets, StatsBase, Statistics

gr(size = (700, 700))

Random.seed!(1)

f1_train = rand(100)
f2_train = rand(100)

p_knn = scatter(f1_train, f2_train,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    title = "k-NN & K-D Tree Demo",
    legend = false,
    color = :green)

X_train = [f1_train f2_train]

X_train_t = permutedims(X_train)

kdtree = KDTree(X_train_t)

#initialise k for k-NN
k = 11

#generate random oint for testing

f1_test = rand()
f2_test = rand()

X_test = [f1_test, f2_test]

scatter!([f1_test], [f2_test],
    color = :yellow, markersize = 8)
#find nearest neighbours using k-NN and k-d Tree

index_knn, distances = knn(kdtree, X_test, k, true)

output = [index_knn distances]

vscodedisplay(output)

#plot the nearest neighbours
f1_knn = [f1_train[i] for i in index_knn]
f2_knn = [f2_train[i] for i in index_knn]

scatter!(f1_knn, f2_knn,
    color = :red, markersize = 8, alpha = 0.5)

#connect test point with nearest neighbours

for i in 1:k
    plot!([f1_test, f1_knn[i]], [f2_test, f2_knn[i]],
        color  = :pink
    )
end

p_knn

#load data
iris = dataset("datasets", "iris")
X= Matrix(iris[:, 1:4])
y = Vector{String}(iris.Species)

#define funvtion
function perclass_splits(y,percent)
    uniq_class = unique(y)
    keep_index = []
    for class in uniq_class
        class_index = findall(y .== class)
        row_index = randsubseq(class_index, percent)
        push!(keep_index, row_index...)
    end
    return keep_index
end

#training and testing

Random.seed!(1)
index_train = perclass_splits(y, 0.73)

index_test = setdiff(1:length(y), index_train)

X_train = X[index_train, :]
X_test = X[index_test, :]
y_train = y[index_train]
y_test = y[index_test]

#transpose
X_train_t = permutedims(X_train)
X_test_t = permutedims(X_test)

#build Tree and run model
kdtree = KDTree(X_train_t)
k = 17
index_knn, distances = knn(kdtree, X_test_t, k, true)
output = [index_test index_knn distances]
vscodedisplay(output)

#post procesing
index_knn_matrix = hcat(index_knn...)
index_knn_matrix_t = permutedims(index_knn_matrix)
vscodedisplay(index_knn_matrix_t)
knn_classes = y_train[index_knn_matrix_t]
vscodedisplay(knn_classes)

#prediction
y_hat = [argmax(countmap(knn_classes[i, :])) for i in 1:length(y_test)]

accuracy = mean(y_hat .== y_test)
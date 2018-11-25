using LinearAlgebra

sigmoid(x) = 1 ./ (1 .+ exp.(-x))

relu(x) = max.(0, x)

function softmax(x)
    if size(x)[1] == 2
        x = x .- maximum(x, dims=2)
        x = exp.(x)
        x = x ./ sum(x, dims=2)
    elseif size(x)[1] == 1
        x = x .- maximum(x, dims=2)
        x = exp.(x) ./ sum(exp.(x))
    end
    return x
end

function cross_entropy_error(y::Vector, t::Vector)
    δ = 1e-7
    -(t ⋅ log(y .+ δ))
end

function cross_entropy_error(y::Matrix, t::Matrix)
    batch_size = size(y, 2)
    δ = 1e-7
    -vecdot(t, log(y .+ δ))
end

function cross_entropy_error(y::Matrix, t::Vector)
    batch_size = size(y, 2)
    δ = 1e-7
    -sum([log(y[t[i] + 1, i]) for i = 1:batch_size] .+ δ) / batch_size
end


x1 = [1 2 3]
x2 = [1 2 3;4 5 6]
println(x1)
println(softmax(x1))
println(x2)
println(softmax(x2))

xo = randn(10)
println(xo)
println(relu(xo))


y = [1 2 3 4 5 6]
t = [1 2 3 1 2 3]

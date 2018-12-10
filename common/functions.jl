using LinearAlgebra

sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))

relu(x) = max.(0, x)

function softmax(x)
    if ndims(x) == 2
        x = x .- maximum(x, dims=2)
        x = exp.(x)
        x = x ./ sum(x, dims=2)
    elseif ndims(x) == 1
        x = x .- maximum(x, dims=2)
        x = exp.(x) ./ sum(exp.(x))
    end
    return x
end


function cross_entropy_error(y::Vector, t::Vector)
    δ = 1e-7
    -t ⋅ log.(y .+ δ)
end

function cross_entropy_error(y::Matrix, t::Matrix)
    batch_size = size(y, 1)
    δ = 1e-7
    _, idx = findmax(t, dims=2)
    nt = reshape(Int.(t[idx]), (batch_size))
    return cross_entropy_error(y, nt)
end

function cross_entropy_error(y::Matrix, t::Vector)
    batch_size = size(y, 1)
    δ = 1e-7
    -sum([log(y[i, t[i]]) for i = 1:batch_size] .+ δ) / batch_size
end

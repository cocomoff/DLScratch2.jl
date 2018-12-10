# -*- coding: utf-8 -*-

module JLlayers

include("functions.jl")


abstract type AbstractLayer
end

export SigmoidLayer, AffineLayer, TwoLayerNet, MatMulLayer, SoftmaxLayer, SoftmaxWithLossLayer
export forward
export backward
export predict
export loss
export gradient
export applygradient!

mutable struct SigmoidLayer{T} <: AbstractLayer
    out::AbstractArray{T}
    function (::Type{SigmoidLayer{T}})() where {T}
        new{T}()
    end
end

function forward(l::SigmoidLayer{T}, x::AbstractArray{T}) where {T}
    l.out = 1 ./ (1 .+ exp.(-x))
    l.out
end

function backward(l::SigmoidLayer{T}, dout::AbstractArray{T}) where {T}
    dout .* (1 .- l.out) .* l.out
end

mutable struct SoftmaxLayer{T} <: AbstractLayer
    out::AbstractArray{T}
    function (::Type{SoftmaxLayer{T}})() where {T}
        new{T}()
    end
end

function forward(l::SoftmaxLayer{T}, x::AbstractArray{T}) where {T}
    l.out = softmax(x)
    l.out
end

function backward(l::SoftmaxLayer{T}, dout::AbstractArray{T}) where {T}
    dx = l.out .* dout
    sumdx = vec(sum(dx, dims=2))
    dx = dx .- l.out * sumdx
    return dx
end

mutable struct SoftmaxWithLossLayer{T} <: AbstractLayer
    loss::T
    y::AbstractArray{T}
    t::AbstractArray{T}
    function (::Type{SoftmaxWithLossLayer{T}})() where {T}
        layer = new{T}()
        layer
    end
end

function forward(l::SoftmaxWithLossLayer{T}, x::AbstractArray{T}, t::AbstractArray{T}) where {T}
    l.t = t
    l.y = softmax(x)
    l.loss = cross_entropy_error(l.y, l.t)
    return l.loss
end


function backward(l::SoftmaxWithLossLayer{T}, dout::T=1.0) where {T}
    dx = copy(l.y)
    dout .* _swlvec(l.y, l.t)
end
@inline _swlvec(y::AbstractArray{T}, t::AbstractVector{T}) where {T} = y .- t
@inline _swlvec(y::AbstractArray{T}, t::AbstractMatrix{T}) where {T} = (y .- t) / size(t)[1]

mutable struct AffineLayer{T} <: AbstractLayer
    W::AbstractMatrix{T}
    b::AbstractVector{T}
    x::AbstractArray{T}
    dW::AbstractMatrix{T}
    db::AbstractVector{T}
    function (::Type{AffineLayer})(W::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
        layer = new{T}()
        layer.W = W
        layer.b = b
        layer.dW = zeros(size(W))
        layer.db = zeros(size(b))
        layer
    end
end

function forward(l::AffineLayer{T}, x::AbstractArray{T}) where {T}
    l.x = x
    x * l.W .+ l.b'
end

function backward(l::AffineLayer{T}, dout::AbstractArray{T}) where {T}
    dx = dout * l.W'
    dW = l.x' * dout
    db = vec(sum(dout, dims=1))
    l.dW .= dW
    l.db .= db
    dx
end

mutable struct TwoLayerNet{T}
    a1l::AffineLayer{T}
    sig::SigmoidLayer{T}
    a2l::AffineLayer{T}
    loss::SoftmaxWithLossLayer{T}
end

function (::Type{TwoLayerNet{T}})(isize::Int, hsize::Int, osize::Int;
        weight_init_std::Float64=1.0) where {T}
#     W1 = weight_init_std * randn(T, isize, hsize)
#     b1 = randn(T, hsize)
#     W2 = weight_init_std * randn(T, hsize, osize)
#     b2 = randn(T, osize)
    W1 = zeros(T, isize, hsize)
    b1 = zeros(T, hsize)
    W2 = zeros(T, hsize, osize)
    b2 = zeros(T, osize)
    a1l = AffineLayer(W1, b1)
    sig = SigmoidLayer{T}()
    a2l = AffineLayer(W2, b2)
    smloss = SoftmaxWithLossLayer{T}()
    TwoLayerNet(a1l, sig, a2l, smloss)
end

function predict(net::TwoLayerNet{T}, x::AbstractArray{T}) where {T}
    a1 = forward(net.a1l, x)
    z1 = forward(net.sig, a1)
    a2 = forward(net.a2l, z1)
    a2
end

function forward(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T}) where {T}
    y = predict(net, x)
    loss = forward(net.loss, y, t)
    return loss
end

struct TwoLayerNetGrads{T}
    W1::AbstractMatrix{T}
    b1::AbstractVector{T}
    W2::AbstractMatrix{T}
    b2::AbstractVector{T}
end

function gradient(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T}) where {T}
    # forward
    yf = forward(net, x, t)
    # println(yf)

    # backward
    dout = one(T)
    dz2 = backward(net.loss, dout)
    da2 = backward(net.a2l, dz2)
    dz1 = backward(net.sig, da2)
    da1 = backward(net.a1l, dz1)
    yf, TwoLayerNetGrads(net.a1l.dW, net.a1l.db, net.a2l.dW, net.a2l.db)
end

function one_hot(x::Array{Float64, 1}; num=3)
    ans = zeros(size(x)[1], num)
    for i in 1:size(x)[1]
        idx = Int(x[i]) + 1
        ans[i, idx] = 1
    end
    return ans
end

function applygradient!(net::TwoLayerNet{T}, grads::TwoLayerNetGrads{T}, learning_rate::T) where {T}
    net.a1l.W -= learning_rate .* grads.W1
    net.a1l.b -= learning_rate .* grads.b1
    net.a2l.W -= learning_rate .* grads.W2
    net.a2l.b -= learning_rate .* grads.b2
end

mutable struct MatMulLayer{T} <: AbstractLayer
    W::AbstractMatrix{T}
    x::AbstractArray{T}
    dW::AbstractMatrix{T}
    function (::Type{MatMulLayer})(W::AbstractMatrix{T}) where {T}
        layer = new{T}()
        layer.W = W
        layer.dW = zeros(size(W))
        layer
    end
end

function forward(l::MatMulLayer{T}, x::AbstractArray{T}) where {T}
    l.x = x
    out = x * l.W
    out
end

function backward(l::MatMulLayer{T}, dout::AbstractArray{T}) where {T}
    dx = dout * l.W'
    dW = l.x' * dout
    l.dW .= dW
    dx
end


end

# -*- coding: utf-8 -*-

module JLlayers

include("functions.jl")


abstract type AbstractLayer
end


export SigmoidLayer, AffineLayer, TwoLayerNet, MatMulLayer, SoftmaxLayer, SoftmaxWithLossLayer
export forward
export backward
export predict

mutable struct SigmoidLayer{T} <: AbstractLayer
    out::AbstractArray{T}
    function (::Type{SigmoidLayer{T}})() where {T}
        new{T}()
    end
end

function forward(l::SigmoidLayer{T}, x::AbstractArray{T}) where {T}
    l.out = 1 ./ 1 .+ exp.(-x)
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
    dout .* _swlvec(l.y .- l.t)
end
@inline _swlvec(y::AbstractArray{T}, t::AbstractVector{T}) where {T} = y .- t
@inline _swlvec(y::AbstractArray{T}, t::AbstractMatrix{T}) where {T} = (y .- t) / size(t)[2]


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
    dW = dout * l.x'
    db = vec(sum(dout, dims=2))
    l.dW .= dW
    l.db .= db
    dx
end

mutable struct TwoLayerNet{T}
    a1l::AffineLayer{T}
    sig::SigmoidLayer{T}
    a2l::AffineLayer{T}
end

function (::Type{TwoLayerNet{T}})(isize::Int, hsize::Int, osize::Int;
        weight_init_std::Float64=0.01) where {T}
    W1 = weight_init_std * randn(T, hsize, isize)
    b1 = zeros(T, hsize)
    W2 = weight_init_std * randn(T, osize, hsize)
    b2 = zeros(T, osize)
    a1l = AffineLayer(W1, b1)
    sig = SigmoidLayer{T}()
    a2l = AffineLayer(W2, b2)
    TwoLayerNet(a1l, sig, a2l)
end

function predict(net::TwoLayerNet{T}, x::AbstractArray{T}) where {T}
    a1 = forward(net.a1l, x)
    z1 = forward(net.sig, a1)
    a2 = forward(net.a2l, z1)
    a2
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

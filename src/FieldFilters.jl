module FieldFilters

using FluidFields, QuadGK

export GaussianFilter, BoxFilter, SpectralCutoff, EyinkFilter

abstract type AbstractFilter end

struct GaussianFilter{T<:Real} <: AbstractFilter
    Δ²::T
end

@inline function (g::GaussianFilter)(k2::Real)
    aux = -g.Δ²/24
    return exp(aux*k2)
end

struct BoxFilter{T<:Real} <: AbstractFilter
    Δ²::T
end

@inline function (b::BoxFilter)(k2::Real)
    aux = sqrt(b.Δ²*k2)/2
    return ifelse(k2 == zero(k2), oneunit(k2), sin(aux)/aux)
end

struct SpectralCutoff{T<:Real} <: AbstractFilter
    Δ²::T
end

@inline function (s::SpectralCutoff)(k2::Real)
    aux = inv(s.Δ²)*π^2
    return k2 < aux 
end

struct EyinkFilter{T} <: AbstractFilter
    Δ²::T
end

Gp(k::Real) = abs(k) <= 1/2 ? exp(-(k^2)/(1/4 - k^2)) : 0.0

function G(k)
    if abs(k) >= 1
        return 0.0
    else
        a,_ = quadgk(x->(Gp(x)*Gp(k-x)),-1,1)
        return a/0.4916904064563633
    end
end

@inline function (e::EyinkFilter)(k2::Real)
    kratio = @fastmath sqrt(π*π/(e.Δ²*k2))
    return G(kratio)
end

end # module

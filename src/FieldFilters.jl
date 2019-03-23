module FieldFilters

using FluidFields, QuadGK

export GaussianFilter, BoxFilter, SpectralCutoffFilter, EyinkFilter, SpectralBarrier, filterfield, filterfield!

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

struct SpectralCutoffFilter{T<:Real} <: AbstractFilter
    Δ²::T
end

@inline function (s::SpectralCutoffFilter)(k2::Real)
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
    kratio = @fastmath sqrt((e.Δ²*k2)/(π*π))/2
    return G(kratio)
end

#@inline esp(k,i,p) = abs(k) <= i ? 1.0 : abs(k) < p ? exp(-((abs(k) - i)^2)/((p)^2 - k^2)) : 0.0

@inline esp(k,i,p,::Val{n}) where n = abs(k) <= i ? 1.0 : abs(k) < p ? exp(-((abs(k) - i)^2)/((p)^n - abs(k)^n)) : 0.

struct SpectralBarrier{T,F} <: AbstractFilter
    Δ²::T
    func::F
end

@inline function (s::SpectralBarrier{T,F})(k2::Real) where {T,F}
    return s.func(@fastmath(sqrt(k2)))
end

SpectralBarrier(Δ2) = (l = π/sqrt(Δ2); SpectralBarrier(Δ2,x->esp(x,l,2*l,Val{1/4}())))

function filterfield!(out::AbstractArray{T,3},inp::AbstractArray{T,3},filt::A,kx::AbstractVector,ky::AbstractVector,kz::AbstractVector) where {T,A<:AbstractFilter}
    Threads.@threads for k in eachindex(kz)
        @inbounds begin
        kz2 = kz[k]^2
        for j in eachindex(ky)
            ky2 = muladd(ky[j],ky[j], kz2)
            for i in eachindex(kx)
                k2 = muladd(kx[i],kx[i],ky2)
                out[i,j,k] = filt(k2)*inp[i,j,k]
            end
        end
        end
    end
end

filterfield!(out::AbstractArray{T,3},inp::AbstractField,fil::AbstractFilter) where {T} = filterfield!(out,inp,fil,inp.kx,inp.ky,inp.kz)
filterfield!(field::AbstractField,fil::AbstractFilter) = filterfield!(field,field,fil)

filterfield(inp::AbstractField,fil::AbstractFilter) = filterfield!(similar(inp),inp,fil)

end # module

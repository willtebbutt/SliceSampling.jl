module SliceSampling

using Random, Distributions
import Distributions: sample

struct EllipticalSliceSampler end

"""
    step(::EllipticalSliceSampler, rng::AbstractRNG, D, L, f::AbstractVector{<:Real})

Perform a single step of Ellipical Slice Sampling.
"""
function step(
    ::EllipticalSliceSampler,
    rng::AbstractRNG,
    D,
    logL,
    f::AbstractVector{<:Real},
)
    # Sample elipse from prior
    ν = rand(rng, D)

    # Sample log likelihood threshold
    u = rand(rng)
    logy = logL(f) + log(u)

    # Draw initial proposal and define bracket
    θ = rand(rng, Uniform(0, 2π))
    θ_min, θ_max = θ - 2π, θ

    while true
        f′ = f .* cos(θ) .+ ν .* sin(θ)
        logLf′ = logL(f′)
        if logLf′ > logy
            return f′, logLf′
        else
            if θ < 0
                θ_min = θ
            else
                θ_max = θ
            end
            θ = rand(rng, Uniform(θ_min, θ_max))
        end
    end
    return nothing
end

function sample(
    ess::EllipticalSliceSampler,
    rng::AbstractRNG,
    D,
    logL,
    f::AbstractVector{<:Real},
    N::Int,
)
    fs = Matrix{eltype(f)}(undef, length(f), N + 1)
    logLs = Vector{eltype(f)}(undef, N + 1)
    fs[:, 1] = f
    logLs[1] = logL(f)
    for n in 1:N
        f′, logLf′ = step(ess, rng, D, logL, fs[:, n])
        fs[:, n + 1] = f′
        logLs[n + 1] = logLf′
    end
    return fs, logLs
end

end # module

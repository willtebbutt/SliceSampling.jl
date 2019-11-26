export SliceSampler, StepperOuter

"""
    StepperOuter{Tw<:Real}

Implements the stepping out procedure with width `w`. See fig 3 in [1] for details.

[1] - Neal, Radford M. "Slice sampling." The annals of statistics 31.3 (2003): 705-767.
"""
struct StepperOuter{Tw<:Real}
    w::Tw
    m::Int
end

function expand(rng::AbstractRNG, s::StepperOuter, logf, x₀::Real, logy::Real)
    w, m = s.w, s.m

    U = rand(rng, Uniform(0, 1))
    L = x₀ - w * U
    R = L + w

    V = rand(rng, Uniform(0, 1))
    J = floor(m * V)
    K = (m - 1) - J

    while J > 0 && logy < logf(L)
        L = L - w
        J = J - 1
    end

    while K > 0 && logy < logf(R)
        R = R + w
        K = K - 1
    end

    return (L, R)
end



"""
    accept(::AbstractRNG, ::StepperOuter, ::Any, ::Real, ::Real, ::Real, ::Tuple{Real, Real})

Any point produced by the stepping out procedure is acceptable. See fig 5 from [1].

[1] - Neal, Radford M. "Slice sampling." The annals of statistics 31.3 (2003): 705-767.
"""
function accept(::AbstractRNG, ::StepperOuter, ::Any, ::Real, ::Real, ::Real, ::Tuple{Real, Real})
    return true
end



"""
    Doubler{Tw<:Real}

Implements the stepping out procedure with width `w`. See fig 4 in [1] for details.

[1] - Neal, Radford M. "Slice sampling." The annals of statistics 31.3 (2003): 705-767.
"""
struct Doubler{Tw<:Real}
    w::Tw
    p::Int
end

function expand(rng::AbstractRNG, s::Doubler, logf, x₀::Real, logy::Real)
    w, p = s.w, s.p

    U = rand(rng, Uniform(0, 1))
    L = x₀ - w * U
    R = L + w
    K = p

    while K > 0 && (logy < logf(L) || logy < logf(R))
        V = rand(rng, Uniform(0, 1))
        if V < 0.5
            L = L - (R - L)
        else
            R = R + (R - L)
        end
        K = K - 1
    end

    return (L, R)
end



"""
    accept(rng::AbstractRNG, logp, x₀::Real, x₁::Real, logy::Real, (L, R)::Tuple{Real, Real})

Implements the accept criterion detailed in fig 6 of [1].

[1] - Neal, Radford M. "Slice sampling." The annals of statistics 31.3 (2003): 705-767.
"""
function accept(
    rng::AbstractRNG,
    doubler::Doubler,
    logp,
    x₀::Real,
    x₁::Real,
    logy::Real,
    (L, R)::Tuple{Real, Real},
)
    throw(error("Not implemented yet."))
end



"""
    shrink(rng::AbstractRNG, logp, x₀::Real, logy::Real, (L, R)::Tuple{Real, Real})

Implements the shrinkage procedure detailed in fig 5 of [1].

[1] - Neal, Radford M. "Slice sampling." The annals of statistics 31.3 (2003): 705-767.
"""
function shrink(
    rng::AbstractRNG,
    ss::SliceSampler,
    logp,
    x₀::Real,
    logy::Real,
    (L, R)::Tuple{Real, Real},
)
    L̄ = L
    R̄ = R

    while true
        U = rand(rng, Uniform(0, 1))
        x₁ = L̄ + U * (R̄ - L̄)
        logf_x₁ = logp(x₁)
        if logy < logf_x₁ && accept(rng, ss.expander, logp, x₀, x₁, logy, (L̄, R̄))
            return x₁, logf_x₁
        end

        if x₁ < x₀
            L̄ = x₁
        else
            R̄ = x₁
        end
    end
end



"""
    SliceSampler{Texpander}

Implements Slice Sampling [1].

[1] - Neal, Radford M. "Slice sampling." The annals of statistics 31.3 (2003): 705-767.
"""
struct SliceSampler{Texpander}
    expander::Texpander
end

function step(
    ss::SliceSampler,
    rng::AbstractRNG,
    logp,
    x₀::Real,
)
    logy = logp(x₀) - rand(rng, Exponential(1))
    L, R = expand(rng, ss.expander, logp, x₀, logy)
    return shrink(rng, ss, logp, x₀, logy, (L, R))
end



"""
    sample(ss::SliceSampler, rng::AbstractRNG, logp, x₀::T, S::Int) where {T<:Real}

Generate `S` samples from the distribution with density proportional to `exp(logp(x))`.
"""
function sample(ss::SliceSampler, rng::AbstractRNG, logp, x₀::T, S::Int) where {T<:Real}

    # Initialise sample storage.
    xs = Vector{T}(undef, S + 1)
    xs[1] = x₀

    # Initialise logp storage.
    logp_x₀ = logp(x₀)
    logps = Vector{typeof(logp_x₀)}(undef, S + 1)
    logps[1] = logp_x₀

    for s in 1:S
        x, logp_x = step(ss, rng, logp, xs[s])
        xs[s + 1] = x
        logps[s + 1] = logp_x
    end
    return xs, logps
end

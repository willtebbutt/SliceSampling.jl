using Test, SliceSampling, Random, Distributions, LinearAlgebra
using SliceSampling: sample, EllipticalSliceSampler

_psd_matrix(A) = Matrix(Symmetric(A * A' + I))

@testset "Gaussian Test Problem" begin

    # Define toy problem
    rng, D = MersenneTwister(123456), 3
    Σ = _psd_matrix(randn(rng, D, D))
    F, σ_noise = MvNormal(Σ), 0.5

    # Generate toy data from problem
    f, ε = rand(rng, F), σ_noise .* randn(rng, D)
    y = f + ε

    # Compute exact posterior
    C = cholesky(Σ + σ_noise^2 * I)
    m′, Σ′ = Σ * (C \ y), Matrix(Symmetric(Σ - Σ * (C \ Σ)))

    # Define the log likelihood
    logL = f->sum(logpdf.(Normal.(f, σ_noise), y))

    # Sample from the posterior using ESS
    fs, _ = sample(EllipticalSliceSampler(), rng, F, logL, rand(rng, F), 10_000_000)
    m′_emp, Σ′_emp = vec(mean(fs; dims=2)), cov(fs; dims=2)
    @test m′ ≈ m′_emp rtol=1e-4
    @test Σ′ ≈ Σ′_emp rtol=1e-2
end

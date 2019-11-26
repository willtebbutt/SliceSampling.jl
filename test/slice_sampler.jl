@testset "slice_sampler" begin
    distributions = [
        Normal(0, 1),
        Normal(-5.0, 1),
        Normal(0.0, 2.0),
        Exponential(1),
        Laplace(0.0, 1.0),
        LogNormal(0.0, 1.0),
    ]
    @testset "$d" for d in distributions
        rng = MersenneTwister(123456)
        S = 100_000_000

        # Generate lots of samples via slice sampling.
        expander = StepperOuter(1.0, 5)
        ss = SliceSampler(expander)
        x₀ = 0.3
        xs, logps = sample(ss, rng, x->logpdf(d, x), x₀, S)

        @test mean(xs) ≈ mean(d) rtol=1e-2 atol=1e-2
        @test std(xs) ≈ std(d) rtol=1e-2 atol=1e-2
    end
end

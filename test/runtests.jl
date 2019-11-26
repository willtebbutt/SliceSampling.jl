using Test, SliceSampling, Random, Distributions, LinearAlgebra
using SliceSampling: sample, EllipticalSliceSampler

_psd_matrix(A) = Matrix(Symmetric(A * A' + I))

@testset "SliceSampling" begin
    include("slice_sampler.jl")
    include("elliptical_slice_sampler.jl")
end

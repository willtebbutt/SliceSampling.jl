module SliceSampling

using Random, Distributions
import Distributions: sample

include("slice_sampler.jl")
include("elliptical_slice_sampler.jl")

end # module

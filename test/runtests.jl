using Distributed
# Thanks to https://stackoverflow.com/a/70895939/5056635 for the exeflags tip.
addprocs(; exeflags="--project=$(Base.active_project())")

@info "nprocs() = $(nprocs())"
import .Threads
@info "nthreads() = $(Threads.nthreads())"

include("test_utilities.jl")
include_everywhere("_models.jl")

@everywhere begin
    using Test
    using Random
    using StableRNGs
    using MLJEnsembles
    using MLJBase
    using ..Models
    using CategoricalArrays
    import Distributions
    using StatisticalMeasures
    import Distributed
end

include("ensembles.jl")
include("serialization.jl")

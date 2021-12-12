module MLJEnsembles

using MLJModelInterface
import MLJModelInterface: predict, fit
import MLJBase # still needed for aggregating measures in oob-estimates of error
using Random
using CategoricalArrays
using CategoricalDistributions
using ComputationalResources
using Distributed
import Distributions
using ProgressMeter
import StatsBase

export EnsembleModel

const MMI = MLJModelInterface

include("ensembles.jl")

end # module

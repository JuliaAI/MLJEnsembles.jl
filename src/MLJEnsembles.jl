module MLJEnsembles

using MLJModelInterface
import MLJModelInterface: predict, fit, save, restore
using Random
using CategoricalArrays
using CategoricalDistributions
using ComputationalResources
using Distributed
import Distributions
using ProgressMeter
import StatsBase
import StatisticalMeasuresBase

export EnsembleModel

const MMI = MLJModelInterface

include("ensembles.jl")
include("serialization.jl")

end # module

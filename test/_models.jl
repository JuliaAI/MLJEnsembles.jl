# If adding models from MLJModels for testing purposes, then do the
# following in the interface file (eg, DecisionTree.jl):

# - change `import ..DecisionTree` to `import DecisionTree`
# - remove wrapping as module

module Models

using MLJBase
import MLJModelInterface: @mlj_model, metadata_model, metadata_pkg
import MLJModelInterface

include("_models/Constant.jl")
include("_models/NearestNeighbors.jl")

end

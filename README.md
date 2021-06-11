# MLJEnsembles.jl 

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/JuliaAI/MLJEnsembles.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJEnsembles.jl/actions)| [![codecov.io](http://codecov.io/github/JuliaAI/MLJEnsembles.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaAI/MLJEnsembles.jl?branch=master) |

A package allowing one to a create a bagged homogeneous ensemble of
machine learning models using the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine
learning framework.


## Installation

No installation is necessary when using MLJ:

```julia
using Pkg
using MLJ
```

For a minimal installation:

```julia
using Pkg
Pkg.add("MLJBase")
Pkg.add("MLJEnsembles")
```

In this case you will also need to load code definining an atomic
model to ensemble. The easiest way to do this is run
`Pkg.add(MLJModels)` and use the `@load` macro.  See the [Loading
Model
Code](https://alan-turing-institute.github.io/MLJ.jl/dev/loading_model_code/)
of the MLJ manual for this and other possibilities. 


## Sample usage

See [Data Science Tutorials](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/ensembles/).


## Documentation

See the [MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/homogeneous_ensembles/#Homegeneous-Ensembles).


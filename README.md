# MLJEnsembles.jl 

[![Build status](https://github.com/JuliaAI/MLJEnsembles.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJEnsembles.jl/actions) [![codecov.io](http://codecov.io/github/JuliaAI/MLJEnsembles.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaAI/MLJEnsembles.jl?branch=master) 

A package to create bagged homogeneous ensembles of
machine learning models using the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine
learning framework.

For combining models in more general ways, see the [Composing
Models](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Composing-Models)
section of the MLJ manual.


## Installation

No installation is necessary when using MLJ, which is installed like this:

```julia
using Pkg
Pkg.add("MLJ")
using MLJ
```

Alternatively, for a "minimal" installation:

```julia
using Pkg
Pkg.add("MLJBase")
Pkg.add("MLJEnsembles")
using MLJBase, MLJEnsembles
```

In this case you will also need to load code defining an atomic model
to ensemble. The easiest way to do this is run `Pkg.add("MLJModels");
using MLJModels` and use the `@load` macro.  See the [Loading Model
Code](https://alan-turing-institute.github.io/MLJ.jl/dev/loading_model_code/)
of the MLJ manual for this and other possibilities.


## Sample usage

See [Data Science Tutorials](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/ensembles/).


## Documentation

See the [MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/homogeneous_ensembles/#Homogeneous-Ensembles).


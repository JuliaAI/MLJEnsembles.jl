# # ENSEMBLES OF FITRESULTS

# Atom is atomic model type, eg, DecisionTree
# R will be the tightest type of the atom fit-results.
mutable struct WrappedEnsemble{R,Atom <: Supervised} <: MLJType
    atom::Atom
    ensemble::Vector{R}
end

# A corner case here is wrapped ensembles of categorical elements (eg,
# ensembles of fitresults for ConstantClassifier). These appear
# because doing comprehension with categorical elements gives
# CategoricalVector instead of Vector, but Vector is required in above
# struct definition.
function WrappedEnsemble(atom, ensemble::AbstractVector{L}) where L
    ensemble_vec = Vector{L}(undef, length(ensemble))
    for k in eachindex(ensemble)
        ensemble_vec[k] = ensemble[k]
    end
    return WrappedEnsemble(atom, ensemble_vec)
end

# to enable trait-based dispatch of predict:
# The following definitions of `predict` function on `WrappedEnsemble`s,
# Xnew` is assumed to be the output of `reformat(atom::Atom, X)` where
# `X` is the external (user-supplied) representation.
function predict(wens::WrappedEnsemble{R,Atom}, atomic_weights, Xnew
                 ) where {R,Atom<:Deterministic}
    predict(wens, atomic_weights, Xnew, Deterministic, target_scitype(Atom))
end

function predict(wens::WrappedEnsemble{R,Atom}, atomic_weights, Xnew
                 ) where {R,Atom<:Probabilistic}
    predict(wens, atomic_weights, Xnew, Probabilistic, target_scitype(Atom))
end

function predict(
    wens::WrappedEnsemble,
    atomic_weights,
    Xnew,
    ::Type{Deterministic},
    ::Type{<:AbstractVector{<:Union{Missing,Finite}}},
)
    # atomic_weights ignored in this case
    ensemble = wens.ensemble
    atom     = wens.atom
    n_atoms = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make this more memory efficient but note that the type of
    # Xnew is unknown (ie, model dependent)
    preds_gen   = (predict(atom, fitresult, Xnew) for fitresult in ensemble)
    predictions = hcat(preds_gen...)

    classes    = levels(predictions)
    n          = size(predictions, 1)
    prediction =
        categorical(vcat([mode(predictions[i,:]) for i in 1:n], classes))[1:n]
    return prediction
end

function predict(
    wens::WrappedEnsemble,
    atomic_weights,
    Xnew,
    ::Type{Deterministic},
    ::Type{<:AbstractVector{<:Union{Missing,Continuous}}},
)
    # considering atomic weights
    ensemble = wens.ensemble
    atom     = wens.atom
    n_atoms  = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make more memory efficient:
    preds_gen   = (atomic_weights[k] * predict(atom, ensemble[k], Xnew)
                    for k in 1:n_atoms)
    predictions = hcat(preds_gen...)
    prediction  = [sum(predictions[i,:]) for i in 1:size(predictions, 1)]

    return prediction
end

function predict(
    wens::WrappedEnsemble,
    atomic_weights,
    Xnew,
    ::Type{Probabilistic},
    ::Type{<:AbstractVector{<:Union{Missing,Finite}}},
)
    ensemble = wens.ensemble
    atom     = wens.atom
    n_atoms  = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make this more memory efficient but note that the type of
    # Xnew is unknown (ie, model dependent):
    # a matrix of probability distributions:
    predictions = [predict(atom, fitresult, Xnew) for fitresult in ensemble]

    # the weighted averages over the ensemble of the discrete pdf's:
    return atomic_weights .* predictions |> sum
end

function predict(
    wens::WrappedEnsemble,
    atomic_weights,
    Xnew,
    ::Type{Probabilistic},
    ::Type{<:AbstractVector{<:Union{Missing,Continuous}}},
)
    ensemble = wens.ensemble
    atom     = wens.atom
    n_atoms  = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make this more memory efficient but note that the type of
    # Xnew is unknown (ie, model dependent):
    # a matrix of probability distributions:
    preds_gen   = (predict(atom, fitresult, Xnew) for fitresult in ensemble)
    predictions = hcat(preds_gen...)

    # TODO: return normal distributions in special case of normal predictions
    # n_rows = size(predictions, 1)
    # # the weighted average over the ensemble of the pdf
    # # means and pdf variances:
    # μs  = [sum([atomic_weights[k]*mean(predictions[i,k])
    #    for k in 1:n_atoms]) for i in 1:n_rows]
    # σ2s = [sum([atomic_weights[k]*var(predictions[i,k])
    #    for k in 1:n_atoms]) for i in 1:n_rows]

    # # a vector of normal probability distributions:
    # prediction = [Distributions.Normal(μs[i], sqrt(σ2s[i])) for i in 1:n_rows]

    prediction = [Distributions.MixtureModel(predictions[i,:], atomic_weights)
                  for i in 1:size(predictions, 1)]

    return prediction

end


# # CORE ENSEMBLE-BUILDING FUNCTIONS

# for when out-of-bag performance estimates are requested:
function get_ensemble_and_indices(atom::Supervised, verbosity, n, n_patterns,
                      n_train, rng, progress_meter, args...)

    ensemble_indices =
        [StatsBase.sample(rng, 1:n_patterns, n_train, replace=false) for i in 1:n]
    ensemble = map(ensemble_indices) do train_rows
        verbosity == 1 && next!(progress_meter)
        verbosity < 2 ||  print("#")
        atom_fitresult, atom_cache, atom_report = fit(
            atom, verbosity - 1, selectrows(atom, train_rows, args...)...
        )
        atom_fitresult
    end
    verbosity < 1 || println()

    return (ensemble, ensemble_indices)

end

# for when out-of-bag performance estimates are not requested:
function get_ensemble(atom::Supervised, verbosity, n, n_patterns,
                      n_train, rng, progress_meter, args...)

    # define generator of training rows:
    if n_train == n_patterns
        # keep deterministic by avoiding re-ordering:
        ensemble_indices = (1:n_patterns for i in 1:n)
    else
        ensemble_indices =
            (StatsBase.sample(rng, 1:n_patterns, n_train, replace=false) for i in 1:n)
    end

    ensemble = map(ensemble_indices) do train_rows
        verbosity == 1 && next!(progress_meter)
        verbosity < 2 ||  print("#")
        atom_fitresult, atom_cache, atom_report = fit(
            atom, verbosity - 1, selectrows(atom, train_rows, args...)...
        )
        atom_fitresult
    end
    verbosity < 1 || println()

    return ensemble

end


# for combining vectors:
_reducer(p, q) = vcat(p, q)
# for combining 2-tuples of vectors:
_reducer(p::Tuple, q::Tuple) = (vcat(p[1], q[1]), vcat(p[2], q[2]))



# # ENSEMBLE MODEL TYPES

mutable struct DeterministicEnsembleModel{Atom<:Deterministic} <: Deterministic
    model::Atom
    atomic_weights::Vector{Float64}
    bagging_fraction::Float64
    rng::Union{Int,AbstractRNG}
    n::Int
    acceleration::AbstractResource
    out_of_bag_measure # TODO: type this
end

mutable struct ProbabilisticEnsembleModel{Atom<:Probabilistic} <: Probabilistic
    model::Atom
    atomic_weights::Vector{Float64}
    bagging_fraction::Float64
    rng::Union{Int, AbstractRNG}
    n::Int
    acceleration::AbstractResource
    out_of_bag_measure
end

const EitherEnsembleModel{Atom} =
    Union{DeterministicEnsembleModel{Atom}, ProbabilisticEnsembleModel{Atom}}

function clean!(model::EitherEnsembleModel)

    if model isa DeterministicEnsembleModel

        ok_target = target_scitype(model.model) <:
            Union{AbstractVector{<:Finite},AbstractVector{<:Continuous}}
        ok_target || error("atomic model has unsupported target_scitype "*
                           "`$(target_scitype(model.model))`. ")
    end

    message = ""

    if model.bagging_fraction > 1 || model.bagging_fraction <= 0
        message = message*"`bagging_fraction` should be "*
        "in the range (0,1]. Reset to 1. "
        model.bagging_fraction = 1.0
    end

    isempty(model.atomic_weights) && return message

    if model isa Deterministic &&
        target_scitype(model.model) <: AbstractVector{<:Finite}
        message = message*"`atomic_weights` will be ignored to "*
            "form predictions, as unsupported for `Finite` targets. "
    else
        total = sum(model.atomic_weights)
        if !(total ≈ 1.0)
            message = message*"atomic_weights should sum to one and are being "*
                "replaced by normalized weights. "
            model.atomic_weights = model.atomic_weights/total
        end
    end

    return message

end


# # USER-FACING CONSTRUCTOR

const ERR_MODEL_UNSPECIFIED = ArgumentError(
"Expecting atomic model as argument. None specified. Use "*
    "`EnsembleModel(model=...)`. ")
const ERR_TOO_MANY_ARGUMENTS = ArgumentError(
    "At most one non-keyword argument, a model, allowed. ")


"""
    EnsembleModel(model,
                  atomic_weights=Float64[],
                  bagging_fraction=0.8,
                  n=100,
                  rng=GLOBAL_RNG,
                  acceleration=CPU1(),
                  out_of_bag_measure=[])

Create a model for training an ensemble of `n` clones of `model`, with
optional bagging. Ensembling is useful if `fit!(machine(atom,
data...))` does not create identical models on repeated calls (ie, is
a stochastic model, such as a decision tree with randomized node
selection criteria), or if `bagging_fraction` is set to a value less
than 1.0, or both.

Here the atomic `model` must support targets with scitype
`AbstractVector{<:Finite}` (single-target classifiers) or
`AbstractVector{<:Continuous}` (single-target regressors).

If `rng` is an integer, then `MersenneTwister(rng)` is the random
number generator used for bagging. Otherwise some `AbstractRNG` object
is expected.

The atomic predictions are optionally weighted according to the vector
`atomic_weights` (to allow for external optimization) except in the
case that `model` is a `Deterministic` classifier, in which case
`atomic_weights` are ignored.

The ensemble model is `Deterministic` or `Probabilistic`, according to
the corresponding supertype of `atom`. In the case of deterministic
classifiers (`target_scitype(atom) <: Abstract{<:Finite}`), the
predictions are majority votes, and for regressors
(`target_scitype(atom)<: AbstractVector{<:Continuous}`) they are
ordinary averages.  Probabilistic predictions are obtained by
averaging the atomic probability distribution/mass functions; in
particular, for regressors, the ensemble prediction on each input
pattern has the type `MixtureModel{VF,VS,D}` from the Distributions.jl
package, where `D` is the type of predicted distribution for `atom`.

Specify `acceleration=CPUProcesses()` for distributed computing, or
`CPUThreads()` for multithreading.

If a single measure or non-empty vector of measures is specified by
`out_of_bag_measure`, then out-of-bag estimates of performance are
written to the training report (call `report` on the trained
machine wrapping the ensemble model).

*Important:* If per-observation or class weights `w` (not to be confused with atomic
weights) are specified when constructing a machine for the ensemble model, as in `mach =
machine(ensemble_model, X, y, w)`, then `w` is used by any measures specified in
`out_of_bag_measure` that support them.

"""
function EnsembleModel(
    args...;
    model=nothing,
    atomic_weights=Float64[],
    bagging_fraction=0.8,
    rng=Random.GLOBAL_RNG,
    n::Int=100,
    acceleration=CPU1(),
    out_of_bag_measure=[]
)

    length(args) < 2 || throw(ERR_TOO_MANY_ARGUMENTS)
    if length(args) === 1
        atom = first(args)
        model === nothing ||
            @warn "Using `model=$atom`. Ignoring specification "*
            "`model=$model`. "
    else
        model === nothing && throw(ERR_MODEL_UNSPECIFIED)
        atom = model
    end

    arguments = (
        atom,
        atomic_weights,
        float(bagging_fraction),
        rng,
        n,
        acceleration,
        out_of_bag_measure
    )

    if atom isa Deterministic
        emodel =  DeterministicEnsembleModel(arguments...)
    elseif atom isa Probabilistic
        emodel = ProbabilisticEnsembleModel(arguments...)
    else
        error("$atom does not appear to be a Supervised model.")
    end

    message = clean!(emodel)
    isempty(message) || @warn message
    return emodel
end


# # THE COMMON FIT AND PREDICT METHODS

function _fit(res::CPU1, func, verbosity, stuff)
    atom, n, n_patterns, n_train, rng, progress_meter, args = stuff
    verbosity < 2 ||  @info "One hash per new atom trained: "
    return func(atom, verbosity, n, n_patterns, n_train, rng, progress_meter, args...)
end

function _fit(res::CPUProcesses, func, verbosity, stuff)
    atom, n, n_patterns, n_train, rng, progress_meter, args = stuff
    if verbosity > 0
        println("Ensemble-building in parallel on $(nworkers()) processors.")
    end

    chunk_size = div(n, nworkers())
    left_over = mod(n, nworkers())

    return @distributed (_reducer) for i = 1:nworkers()
        if i != nworkers()
            func(atom, 0, chunk_size, n_patterns, n_train, rng, progress_meter, args...)
        else
            func(
                atom,
                0,
                chunk_size + left_over,
                n_patterns,
                n_train,
                rng,
                progress_meter,
                args...,
            )
        end
    end
end

# Create thread safe version of RNGs.
# Random._GLOBAL_RNG() and Random.default_rng() are threadsafe by default_rng
# as they have thread local state from julia >=1.3<=1.6 and task local state Julia >=1.7   
threadsafe_rng(rng::typeof(Random.default_rng())) = rng
if isdefined(Random, :_GLOBAL_RNG)
    threadsafe_rng(rng::Random._GLOBAL_RNG) = rng
end
threadsafe_rng(rng) = deepcopy(rng)

function _fit(res::CPUThreads, func, verbosity, stuff)
    atom, n, n_patterns, n_train, rng, progress_meter, args = stuff
    if verbosity > 0
        println("Ensemble-building in parallel on $(Threads.nthreads()) threads.")
    end
    
    nthreads = Threads.nthreads()
    
    if nthreads == 1
        return  _fit(CPU1(), func, verbosity, stuff)
    end

    chunk_size = div(n, nthreads)
    left_over = mod(n, nthreads)
    resvec = Vector(undef, nthreads) # FIXME: Make this type-stable?

    @sync begin
        for i in 1:nthreads-1
            Threads.@spawn(
                resvec[i] = func(
                    atom,
                    0,
                    chunk_size,
                    n_patterns,
                    n_train,
                    threadsafe_rng(rng),
                    progress_meter,
                    args...
                )
            )
        end
        Threads.@spawn(
            resvec[nthreads] = func(
                    atom,
                    0,
                    chunk_size + left_over,
                    n_patterns,
                    n_train,
                    threadsafe_rng(rng),
                    progress_meter,
                    args...
            )
        )
    end

    return reduce(_reducer, resvec)
end

# for subsampling weights, which could be `nothing`, per-observation weights, or
# class_weights:
_view(class_weights::AbstractDict, rows) = class_weights
_view(::Nothing, rows) = nothing
_view(weights, rows) = view(weights, rows)

function MMI.fit(
    model::EitherEnsembleModel{Atom}, verbosity::Int, args...
) where Atom<:Supervised

    X = args[1]
    y = args[2]
    if length(args) == 3
        w = args[3]
    else
        w = nothing
    end

    # model specific reformated args is required for calling
    # `fit`/`predict` on the `atom` model.
    atom = model.model
    atom_specific_args = MMI.reformat(atom, args...)
    atom_specific_X = atom_specific_args[1]

    acceleration = model.acceleration
    if acceleration isa CPUProcesses && nworkers() == 1
        acceleration = CPU1()
    end

    # we wrap the measures in `robust_measure` so they can be called with weights, even
    # when they don't support them, and just ignore them silently.
    if model.out_of_bag_measure isa Vector
        out_of_bag_measure =
            StatisticalMeasuresBase.robust_measure.(model.out_of_bag_measure)
    else
        out_of_bag_measure =
            [StatisticalMeasuresBase.robust_measure(model.out_of_bag_measure),]
    end

    if model.rng isa Integer
        rng = MersenneTwister(model.rng)
    else
        rng = model.rng
    end

    n = model.n
    n_patterns = nrows(y)
    n_train = round(Int, floor(model.bagging_fraction*n_patterns))

    progress_meter = Progress(
        n,
        dt=0.5,
        desc="Training ensemble: ",
        barglyphs=BarGlyphs("[=> ]"),
        barlen=50,
        color=:yellow
    )

    stuff = (atom, n, n_patterns, n_train, rng, progress_meter, atom_specific_args)
    if !isempty(out_of_bag_measure)
        ensemble, ensemble_indices = _fit(
            acceleration, get_ensemble_and_indices, verbosity, stuff
        )
    else
        ensemble = _fit(acceleration, get_ensemble, verbosity, stuff)
    end

    fitresult = WrappedEnsemble(model.model, ensemble)

    if !isempty(out_of_bag_measure)

        measurements=zeros(length(ensemble),length(out_of_bag_measure))
        for i= 1:length(ensemble)
            #oob indices
            ooB_indices=  setdiff(1:n_patterns, ensemble_indices[i])
            if isempty(ooB_indices)
                error("Empty out-of-bag sample. "*
                      "Data size too small or "*
                      "bagging_fraction too close to 1.0. ")
            end
            yhat = predict(
                atom,
                ensemble[i],
                selectrows(atom, ooB_indices, atom_specific_X)...,
            )
            Xtest = selectrows(X, ooB_indices)
            ytest = selectrows(y, ooB_indices)

            # this could be class weights OR per-observation weights, OR `nothing`:
            wtest = _view(w, ooB_indices)

            for k in eachindex(out_of_bag_measure)
                m = out_of_bag_measure[k]
                s = m(yhat, ytest, wtest)
                measurements[i,k] = s
            end
        end

        # aggregate measurements across the ensembles:
        aggregated_measurements = map(eachindex(out_of_bag_measure)) do k
            StatisticalMeasuresBase.aggregate(
                measurements[:,k],
                mode=StatisticalMeasuresBase.external_aggregation_mode(
                    out_of_bag_measure[k],
                )
            )
        end

        names = Symbol.(string.(out_of_bag_measure))

    else
        aggregated_measurements = missing
    end

    report=(
        measures=out_of_bag_measure,
        oob_measurements=aggregated_measurements,
    )
    cache = deepcopy(model)

    return fitresult, cache, report

end

# if n is only parameter that changes, we just append to the existing
# ensemble, or truncate it:
function MMI.update(model::EitherEnsembleModel,
                                  verbosity::Int, fitresult, old_model, args...)

    n = model.n

    if MMI.is_same_except(model.model, old_model.model,
                              :n, :atomic_weights, :acceleration)
        if n > old_model.n
            verbosity < 1 ||
                @info "Building on existing ensemble of length $(old_model.n)"
            model.n = n - old_model.n # temporarily mutate the model
            wens, model_copy, report = fit(model, verbosity, args...)
            append!(fitresult.ensemble, wens.ensemble)
            model.n = n         # restore model
            model_copy.n = n    # new copy of the model
        else
            verbosity < 1 || @info "Truncating existing ensemble."
            fitresult.ensemble = fitresult.ensemble[1:n]
            model_copy = deepcopy(model)
        end
        cache, report = model_copy, NamedTuple()
        return fitresult, cache, report
    else
        return fit(model, verbosity, args...)
    end

end

function MMI.predict(model::EitherEnsembleModel, fitresult, Xnew)

    n = model.n
    if isempty(model.atomic_weights)
        atomic_weights = fill(1/n, n)
    else
        length(model.atomic_weights) == n ||
            error("Ensemble size and number of atomic_weights not the same.")
        atomic_weights = model.atomic_weights
    end
    atom = model.model
    return predict(fitresult, atomic_weights, reformat(atom, Xnew)...)
end


# # METADATA

# Note: input and target traits are inherited from atom

MMI.load_path(::Type{<:ProbabilisticEnsembleModel}) =
    "MLJ.ProbabilisticEnsembleModel"
MMI.load_path(::Type{<:DeterministicEnsembleModel}) =
    "MLJ.DeterministicEnsembleModel"

MMI.is_wrapper(::Type{<:EitherEnsembleModel}) = true
MMI.supports_weights(::Type{<:EitherEnsembleModel{Atom}}) where Atom =
    MMI.supports_weights(Atom)
MMI.package_name(::Type{<:EitherEnsembleModel}) = "MLJEnsembles"
MMI.package_uuid(::Type{<:EitherEnsembleModel}) =
    "50ed68f4-41fd-4504-931a-ed422449fee0"
MMI.package_url(::Type{<:EitherEnsembleModel}) =
    "https://github.com/JuliaAI/MLJEnsembles.jl"
MMI.is_pure_julia(::Type{<:EitherEnsembleModel{Atom}}) where Atom =
    MMI.is_pure_julia(Atom)
MMI.input_scitype(::Type{<:EitherEnsembleModel{Atom}}) where Atom =
    MMI.input_scitype(Atom)
MMI.target_scitype(::Type{<:EitherEnsembleModel{Atom}}) where Atom =
    MMI.target_scitype(Atom)

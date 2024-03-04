module TestSerialization

using Test
using MLJEnsembles
using MLJBase
using ..Models
using Serialization

function test_args(mach)
    # Check source nodes are empty if any
    for arg in mach.args
        if arg isa Source
            @test arg == source()
        end
    end
end

test_data(mach) = @test all([:old_rows, :data, :resampled_data, :cache]) do field
    !isdefined(mach, field) || isnothing(getfield(mach, field))
end

function generic_tests(mach₁, mach₂)
    test_args(mach₂)
    test_data(mach₂)
    @test mach₂.state == -1
    for field in (:frozen, :model, :old_model, :old_upstream_state)
        @test getfield(mach₁, field) == getfield(mach₂, field)
    end
end

@testset "Test serializable Ensemble machine" begin
    filename = "ensemble_mach.jls"
    X, y = make_regression(100)
    model = EnsembleModel(model=KNNRegressor())
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    smach = MLJBase.serializable(mach)
    @test smach.report === mach.report
    generic_tests(mach, smach)
    @test smach.fitresult isa MLJEnsembles.WrappedEnsemble
    @test smach.fitresult.atom == model.model

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    MLJBase.restore!(smach)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test fitted_params(smach) isa NamedTuple
    @test report(smach).measures == report(mach).measures
    @test report(smach).oob_measurements isa Missing
    @test report(mach).oob_measurements isa Missing

    rm(filename)

    # End to end
    MLJBase.save(filename, mach)
    smach = machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm(filename)
end

# define a supervised model with ephemeral `fitresult`, but which overcomes this by
# overloading `save`/`restore`:
thing = []
struct EphemeralRegressor <: Deterministic end
function MLJBase.fit(::EphemeralRegressor, verbosity, X, y)
    # if I serialize/deserialized `thing` then `view` below changes:
    view = objectid(thing)
    fitresult = (thing, view, mean(y))
    return fitresult, nothing, NamedTuple()
end
function MLJBase.predict(::EphemeralRegressor, fitresult, X)
    thing, view, μ = fitresult
    return view == objectid(thing) ? fill(μ, nrows(X)) :
        throw(ErrorException("dead fitresult"))
end
MLJBase.target_scitype(::Type{<:EphemeralRegressor}) = AbstractVector{Continuous}
function MLJBase.save(::EphemeralRegressor, fitresult)
    thing, _, μ = fitresult
    return (thing, μ)
end
function MLJBase.restore(::EphemeralRegressor, serialized_fitresult)
    thing, μ = serialized_fitresult
    view = objectid(thing)
    return (thing, view, μ)
end

@testset "serialization for atomic models with non-persistent fitresults" begin
    # https://github.com/alan-turing-institute/MLJ.jl/issues/1099
    X, y = (; x = rand(10)), fill(42.0, 3)
    ensemble = EnsembleModel(
        EphemeralRegressor(),
        bagging_fraction=0.7,
        n=2,
    )
    mach = machine(ensemble, X, y)
    fit!(mach, verbosity=0)
    io = IOBuffer()
    MLJBase.save(io, mach)
    seekstart(io)
    mach2 = machine(io)
    close(io)
    @test MLJBase.predict(mach2, (; x = rand(2))) ≈ fill(42.0, 2)
end

end

true

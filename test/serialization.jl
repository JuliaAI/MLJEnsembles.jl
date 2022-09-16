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

test_data(mach) = all([:old_rows, :data, :resampled_data, :cache]) do field
    @test !isdefined(mach, field) || isnothing(getfield(mach, field))
end

function generic_tests(mach₁, mach₂)
    test_args(mach₂)
    test_data(mach₂)
    @test mach₂.state == -1
    for field in (:frozen, :model, :old_model, :old_upstream_state, :fit_okay)
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

end

true

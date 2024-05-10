using Test

using ComputationalResources

macro testset_accelerated(name::String, var, ex)
    testset_accelerated(name, var, ex)
end
macro testset_accelerated(name::String, var, opts::Expr, ex)
    testset_accelerated(name, var, ex; eval(opts)...)
end
function testset_accelerated(name::String, var, ex; exclude=[])
    final_ex = quote
       local $var = CPU1()
        @testset $name $ex
    end

    resources = AbstractResource[CPUProcesses(), CPUThreads()]

    for res in resources
        if any(x->typeof(res)<:x, exclude)
            push!(final_ex.args, quote
               local $var = $res
               @testset $(name*" ($(typeof(res).name))") begin
                   @test_broken false
               end
            end)
        else
            push!(final_ex.args, quote
               local $var = $res
               @testset $(name*" ($(typeof(res).name))") $ex
            end)
        end
    end
    # preserve outer location if possible
    if ex isa Expr && ex.head === :block && !isempty(ex.args) &&
        ex.args[1] isa LineNumberNode
        final_ex = Expr(:block, ex.args[1], final_ex)
    end
    return esc(final_ex)
end

function include_everywhere(filepath)
    include(filepath) # Load on Node 1 first, triggering any precompile
    if nprocs() > 1
        fullpath = joinpath(@__DIR__, filepath)
        @sync for p in workers()
            @async remotecall_wait(include, p, fullpath)
        end
    end
end

include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "output.jl"))
using Dates
using Random

const ALGORITHMS = ["L-BFGS", "Nelder-Mead", "Surfit"]
const ALGORITHM_RUNNERS = Dict(
    "L-BFGS" => run_lbfgs,
    "Nelder-Mead" => run_nelder_mead,
    "Surfit" => run_surfit,
)

function benchmark_case(case, scenario; algorithms=ALGORITHMS)
    reference = run_levenberg_marquardt(case, scenario)
    results = BenchmarkResult[reference]
    unknown = setdiff(algorithms, keys(ALGORITHM_RUNNERS))
    isempty(unknown) || throw(ArgumentError("unknown algorithms: $(join(unknown, ", "))"))
    for algorithm in ALGORITHMS
        algorithm in algorithms || continue
        push!(results, ALGORITHM_RUNNERS[algorithm](case, scenario))
    end
    comparison_rows(results)
end

function option_value(args, name, default)
    prefix = "--$(name)="
    argument = findfirst(arg -> startswith(arg, prefix), args)
    argument === nothing ? default : split(args[argument], '='; limit=2)[2]
end

function print_help()
    println("Usage: julia --project=. benchmarks/run_benchmark.jl [output_path] [options]")
    println()
    println("Options:")
    println("  --models=e1,g1e1")
    println("  --scenarios=default,perturbed,widened")
    println("  --algorithms=L-BFGS,Nelder-Mead,Surfit")
    println("  --seed=0")
    println("    (Levenberg-Marquardt is always run as the reference.)")
    println("  --help")
end

function main(args=ARGS)
    "--help" in args && return print_help()
    positional_args = filter(arg -> !startswith(arg, "--"), args)
    data_path = joinpath(@__DIR__, "data", "Zsac_FLUPS_spectra_params.txt")
    output_path = get(positional_args, 1, joinpath(@__DIR__, "results", "benchmark.csv"))
    model_filter = split(option_value(args, "models", "e1,g1e1"), ',')
    scenario_filter = split(option_value(args, "scenarios", "default,perturbed,widened"), ',')
    algorithm_filter = split(option_value(args, "algorithms", join(ALGORITHMS, ',')), ',')
    seed = parse(Int, option_value(args, "seed", "0"))
    Random.seed!(seed)
    rows = NamedTuple[]
    for entry in benchmark_cases(data_path)
        entry.case.name in model_filter || continue
        for scenario in entry.scenarios
            scenario.name in scenario_filter || continue
            append!(rows, benchmark_case(entry.case, scenario; algorithms=algorithm_filter))
        end
    end
    metadata = benchmark_metadata(seed=seed)
    write_results(output_path, add_metadata(rows, metadata))
    println("Wrote $(length(rows)) results to $(output_path)")
    rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
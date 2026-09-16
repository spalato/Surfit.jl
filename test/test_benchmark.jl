using Test

include("../benchmarks/core.jl")
include("../benchmarks/cases.jl")
include("../benchmarks/output.jl")

@testset "benchmark core" begin
    model(x, slope, intercept) = slope .* x .+ intercept
    x = [1.0, 2.0, 3.0]
    y = model(x, 2.0, 1.0)
    case = BenchmarkCase("linear", model, x, y)
    scenario = BenchmarkScenario("default", [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0])

    nm = run_nelder_mead(case, scenario)
    lbfgs = run_lbfgs(case, scenario)
    surfit_result = run_surfit(case, scenario)
    lm = run_levenberg_marquardt(case, scenario)

    @test nm.status == :success
    @test lbfgs.status == :success
    @test surfit_result.status == :success
    @test lm.status == :success
    @test nm.model_calls > 0
    @test lbfgs.model_calls > 0
    @test nm.solver_f_calls !== nothing
    @test lbfgs.solver_f_calls !== nothing
    @test lbfgs.solver_g_calls !== nothing
    @test lm.solver_f_calls > 0
    @test lm.solver_g_calls === nothing
    @test surfit_result.solver_f_calls === nothing
    @test isapprox(nm.objective_value, 0.0; atol=1e-6)
    @test isapprox(lbfgs.objective_value, 0.0; atol=1e-6)
    @test isapprox(lm.objective_value, 0.0; atol=1e-6)
    @test isapprox(surfit_result.objective_value, 0.0; atol=1e-4)
    @test result_row(lbfgs).algorithm == "L-BFGS"

    nondifferentiable_model(x, slope, intercept) = abs(slope) .* x .+ intercept
    nondifferentiable_case = BenchmarkCase(
        "nondifferentiable",
        nondifferentiable_model,
        x,
        y,
    )
    nondifferentiable_scenario = BenchmarkScenario(
        "nondifferentiable-start",
        [0.0, 0.0],
        [-5.0, -5.0],
        [5.0, 5.0],
    )
    nondifferentiable_lm = run_levenberg_marquardt(
        nondifferentiable_case,
        nondifferentiable_scenario,
    )
    @test nondifferentiable_lm.status == :success
    @test nondifferentiable_lm.solver_f_calls > 0

    rows = comparison_rows([lm, lbfgs, nm, surfit_result])
    @test length(rows) == 4
    @test rows[1].algorithm == "Levenberg-Marquardt"
    @test rows[1].valid
    @test rows[2].reference_status == "success"
    @test rows[2].reference_algorithm == "Levenberg-Marquardt"
    @test rows[2].valid
    @test rows[2].algorithm == "L-BFGS"

    alternate_rows = comparison_rows([lbfgs, nm]; reference_algorithm="L-BFGS")
    @test all(row.reference_algorithm == "L-BFGS" for row in alternate_rows)
    @test alternate_rows[1].valid

    variants = scenario_variants("default", [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0])
    @test [scenario.name for scenario in variants] == ["default", "perturbed", "widened"]
    @test all(validate_scenario(scenario) === nothing for scenario in variants)

    mktemp() do path, io
        close(io)
        writedlm(path, [1.0 0.0 3.0; 2.0 0.0 5.0; 3.0 0.0 7.0])
        cases = benchmark_cases(path)
        @test [entry.case.name for entry in cases] == ["e1", "g1e1"]
        @test length(cases[1].scenarios) == 3
        @test cases[1].case.model([1.0, 2.0], 2.0, 1.0, 0.0) ≈ 2.0 .* exp.(-[1.0, 2.0])
    end

    failing_model(x, parameters...) = error("synthetic model failure")
    failing_case = BenchmarkCase("failing", failing_model, [1.0, 2.0], [1.0, 2.0])
    failed = run_nelder_mead(failing_case, scenario)
    @test failed.status == :failed
    @test isnan(failed.objective_value)
    @test failed.error_message !== nothing

    mktempdir() do directory
        metadata = benchmark_metadata(seed=123)
        @test metadata.seed == 123
        @test metadata.julia_version == string(VERSION)
        rows = add_metadata(rows, metadata)
        path = write_results(joinpath(directory, "results.csv"), rows)
        table = read(path, String)
        @test occursin("Levenberg-Marquardt", table)
        @test occursin("model_calls", table)
        @test occursin("julia_version", table)
        @test occursin("seed", table)
        @test !occursin("data_path", table)
        @test !occursin("selected_models", table)
        @test !occursin("selected_scenarios", table)
        @test !occursin("selected_algorithms", table)
    end

    @test_throws ArgumentError validate_scenario(
        BenchmarkScenario("invalid", [0.0], [1.0], [0.0])
    )
end
using LinearAlgebra
using LsqFit
using Optim
using Surfit: surfit

const SURFIT_X_TOL = 1e-8
const SURFIT_F_TOL = 1e-10
const SURFIT_F_CALLS = 100
const SURFIT_INITIAL_SAMPLES = 25
const VALUE_ATOL = 1e-8
const VALUE_RTOL = 1e-6
const PARAMETER_ATOL = 1e-3
const PARAMETER_RTOL = 1e-5

struct BenchmarkCase{F, TX, TY}
    name::String
    model::F
    x::TX
    y::TY
end

struct BenchmarkScenario{T <: AbstractVector}
    name::String
    guess::T
    lower::T
    upper::T
end

mutable struct CountedModel{F, TX}
    model::F
    x::TX
    calls::Int
end

function (counted::CountedModel)(parameters...)
    counted.calls += 1
    counted.model(counted.x, parameters...)
end

struct BenchmarkResult{T <: AbstractVector}
    algorithm::String
    model::String
    scenario::String
    status::Symbol
    parameters::T
    objective_value::Float64
    model_calls::Int
    solver_f_calls::Union{Nothing, Int}
    solver_g_calls::Union{Nothing, Int}
    elapsed_seconds::Float64
    error_message::Union{Nothing, String}
end

function validate_scenario(scenario::BenchmarkScenario)
    length(scenario.guess) == length(scenario.lower) == length(scenario.upper) ||
        throw(ArgumentError("guess and bounds must have equal lengths"))
    all(isfinite, scenario.guess) || throw(ArgumentError("guess must be finite"))
    all(isfinite, scenario.lower) || throw(ArgumentError("lower bounds must be finite"))
    all(isfinite, scenario.upper) || throw(ArgumentError("upper bounds must be finite"))
    all(scenario.lower .< scenario.upper) || throw(ArgumentError("lower bounds must be less than upper bounds"))
    all(scenario.lower .<= scenario.guess .<= scenario.upper) ||
        throw(ArgumentError("guess must be inside the bounds"))
    nothing
end

function scenario_variants(name, guess, lower, upper)
    span = upper .- lower
    perturbed_guess = clamp.(guess .+ 0.1 .* span, lower .+ 0.01 .* span, upper .- 0.01 .* span)
    widened_lower = lower .- 0.25 .* span
    widened_upper = upper .+ 0.25 .* span
    [
        BenchmarkScenario(name, guess, lower, upper),
        BenchmarkScenario("perturbed", perturbed_guess, lower, upper),
        BenchmarkScenario("widened", guess, widened_lower, widened_upper),
    ]
end

function _result(algorithm, case, scenario, counted, parameters, objective_value,
                 solver_f_calls, solver_g_calls, elapsed_seconds, status, error_message)
    BenchmarkResult(
        algorithm,
        case.name,
        scenario.name,
        status,
        collect(parameters),
        Float64(objective_value),
        counted.calls,
        solver_f_calls,
        solver_g_calls,
        elapsed_seconds,
        error_message,
    )
end

function _run(optimizer, algorithm, case, scenario)
    validate_scenario(scenario)
    counted = CountedModel(case.model, case.x, 0)
    residuals(parameters) = case.y .- counted(parameters...)
    objective(parameters) = sum(abs2, residuals(parameters))
    started = time()
    try
        result = optimizer(objective, counted, scenario)
        elapsed = time() - started
        _result(
            algorithm,
            case,
            scenario,
            counted,
            result.parameters,
            objective(result.parameters),
            result.solver_f_calls,
            result.solver_g_calls,
            elapsed,
            result.status,
            nothing,
        )
    catch error
        elapsed = time() - started
        fallback_value = try
            Float64(objective(scenario.guess))
        catch
            NaN
        end
        _result(
            algorithm,
            case,
            scenario,
            counted,
            scenario.guess,
            fallback_value,
            nothing,
            nothing,
            elapsed,
            :failed,
            sprint(showerror, error),
        )
    end
end

function run_nelder_mead(case, scenario; options=Optim.Options())
    _run("Nelder-Mead", case, scenario) do objective, _, scenario
        result = optimize(objective, scenario.guess, NelderMead(), options)
        (
            parameters=Optim.minimizer(result),
            solver_f_calls=Optim.f_calls(result),
            solver_g_calls=nothing,
            status=:success,
        )
    end
end

function run_lbfgs(case, scenario; options=Optim.Options())
    _run("L-BFGS", case, scenario) do objective, _, scenario
        result = optimize(
            objective,
            scenario.lower,
            scenario.upper,
            scenario.guess,
            Fminbox(LBFGS()),
            options,
        )
        (
            parameters=Optim.minimizer(result),
            solver_f_calls=Optim.f_calls(result),
            solver_g_calls=Optim.g_calls(result),
            status=:success,
        )
    end
end

function run_levenberg_marquardt(case, scenario; autodiff=:finiteforward)
    _run("Levenberg-Marquardt", case, scenario) do _, counted, scenario
        residual_calls = Ref(0)
        lm_model(_, parameters) = begin
            residual_calls[] += 1
            counted(parameters...)
        end
        fit = curve_fit(
            lm_model,
            case.x,
            case.y,
            scenario.guess;
            lower=scenario.lower,
            upper=scenario.upper,
            autodiff=autodiff,
        )
        (
            parameters=coef(fit),
            solver_f_calls=residual_calls[],
            solver_g_calls=nothing,
            status=fit.converged ? :success : :failed,
        )
    end
end

function run_surfit(case, scenario)
    _run("Surfit", case, scenario) do _, counted, scenario
        vector_model(_, parameters...) = counted(parameters...)
        parameters, objective_value, _, _ = surfit(
            vector_model,
            case.x,
            case.y,
            scenario.guess,
            scenario.lower,
            scenario.upper,
            SURFIT_X_TOL,
            SURFIT_F_TOL,
            SURFIT_F_CALLS,
            SURFIT_INITIAL_SAMPLES,
        )
        (
            parameters=parameters,
            solver_f_calls=nothing,
            solver_g_calls=nothing,
            status=:success,
        )
    end
end

function result_row(result::BenchmarkResult)
    (
        algorithm=result.algorithm,
        model=result.model,
        scenario=result.scenario,
        status=String(result.status),
        parameters=result.parameters,
        objective_value=result.objective_value,
        model_calls=result.model_calls,
        solver_f_calls=result.solver_f_calls,
        solver_g_calls=result.solver_g_calls,
        elapsed_seconds=result.elapsed_seconds,
        error_message=result.error_message,
    )
end

function comparison_rows(results::AbstractVector{<:BenchmarkResult};
                         reference_algorithm="Levenberg-Marquardt")
    references = filter(result -> result.algorithm == reference_algorithm, results)
    length(references) == 1 || throw(ArgumentError("expected exactly one $(reference_algorithm) result"))
    reference = only(references)
    reference_valid = reference.status == :success

    map(results) do result
        value_error = result.objective_value - reference.objective_value
        parameter_error = norm(result.parameters - reference.parameters)
        value_valid = isapprox(
            result.objective_value,
            reference.objective_value;
            atol=VALUE_ATOL,
            rtol=VALUE_RTOL,
        )
        parameter_valid = isapprox(
            result.parameters,
            reference.parameters;
            atol=PARAMETER_ATOL,
            rtol=PARAMETER_RTOL,
        )
        row = result_row(result)
        merge(
            row,
            (
                reference_status=String(reference.status),
                reference_algorithm=reference_algorithm,
                reference_objective_value=reference.objective_value,
                value_error=value_error,
                parameter_error=parameter_error,
                valid=reference_valid && result.status == :success && value_valid && parameter_valid,
            ),
        )
    end
end
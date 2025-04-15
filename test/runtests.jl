using Test
using TestItems
using Surrogates
using LinearAlgebra
#include("../src/optimisation.jl")
using Surfit: surfit_scalar, residual_vs

@testitem "surfit_scalar basic tests" begin
    model(t, a, b) = a * exp.(-b * t) # Example model
    t = 0:0.1:10                     # Example time data
    y_exp = model.(t, 2.0, 0.5) .+ 0.1 .* randn(length(t)) # Add noise to experimental data
    guess = [1.8, 0.55]              # Initial guess
    lb = [0.5, 0.01]                 # Lower bounds
    ub = [3.0, 1.0]                  # Upper bounds
    x_tol = 1e-6                     # Tolerance for x
    f_tol = 1e-6                     # Tolerance for function value
    f_calls = 200                    # Maximum function calls
    initsamp = 25                    # Initial samples
    frozen_t(a, b) = model.(t, a, b) # Freeze independant variable
    resid = resid_vs(y_exp, frozen_t) # residual function p -> resid
    ssq = ssq_of(resid)              # ssq function p-> ssq
    popt, pval, samples, values = surfit_scalar(ssq, guess, lb, ub, x_tol, f_tol, f_calls, initsamp)
    # test popt has the same type as guess
    @test typeof(popt) == typeof(guess)
    # test pval is a scalar
    @test typeof(pval) == Float64
    @test length(samples) > initsamp
    @test length(values) == length(samples)
end


@testitem "Optimization Smoke testing." begin
    include("./test_optimisation.jl")
end

@testitem "Datastore tests" begin
    include("./tests_datastore.jl")
end
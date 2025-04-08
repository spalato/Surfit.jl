using Test
using Surrogates
using LinearAlgebra
include("../src/optimisation.jl")

@testset "surrogatefit basic tests" begin
    model(t, a, b) = a * exp.(-b * t) # Example model
    t = 0:0.1:10                     # Example time data
    y_exp = model.(t, 2.0, 0.5)      # Example experimental data
    guess = [1.0, 0.1]               # Initial guess
    lb = [0.5, 0.01]                 # Lower bounds
    ub = [3.0, 1.0]                  # Upper bounds
    x_tol = 1e-6                     # Tolerance for x
    f_tol = 1e-6                     # Tolerance for function value
    f_calls = 50                     # Maximum function calls
    initsamp = 10                    # Initial samples

    @testset "Test surrogatefit execution" begin
        result, samples, values = surrogatefit(model, t, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, initsamp)
        @test typeof(result) == NamedTuple
        @test length(samples) > 0
        @test length(values) == length(samples)
    end
end

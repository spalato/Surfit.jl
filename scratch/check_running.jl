using Test
using Surrogates
using LinearAlgebra
using surfit

function check_running()
    model(t, a, b) = a * exp.(-b * t) # Example model
    t = 0:0.1:10                     # Example time data
    y_exp = model.(t, 2.0, 0.5)      # Example experimental data
    guess = [2.0, 0.5]               # Initial guess
    lb = [0.5, 0.01]                 # Lower bounds
    ub = [3.0, 1.0]                  # Upper bounds
    x_tol = 1e-6                     # Tolerance for x
    f_tol = 1e-6                     # Tolerance for function value
    f_calls = 100                    # Maximum function calls
    initsamp = 20                    # Initial samples

    popt, pval, sample, values = surrogatefit(model, t, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, initsamp)
    @infiltrate

end

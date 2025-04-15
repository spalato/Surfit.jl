using Test
using Surfit
using HDF5

@testset "Integration test for surfit and stored" begin
    # Create a temporary HDF5 file
    temp_h5_file = "temp_test_store.h5"
    h5open(temp_h5_file, "w") do store
        # Define a simple model function
        model(t, a, b) = a .* exp.(-b .* t)

        # Generate synthetic data
        t = collect(0:0.1:2)
        y_exp = model.(t, 2.0, 0.5) .+ 0.1 .* randn(length(t))

        # Define optimization parameters
        guess = [1.8, 0.55]
        lb = [0.5, 0.01]
        ub = [3.0, 1.0]
        x_tol = 1e-6
        f_tol = 1e-6
        f_calls = 200

        # Use the stored function
        stored_model = stored(model, store)

        # Perform surrogate optimization
        popt, pval, sampl, sampl_val = surfit(stored_model, t, y_exp, guess, lb, ub, x_tol, f_tol, f_calls)

        # Validate results
        @test typeof(popt) == typeof(guess)
        @test typeof(pval) == Float64
        @test length(sampl) > 0
        @test length(sampl) == length(sampl_val)
    end

    # Clean up temporary file
    rm(temp_h5_file, force=true)
end
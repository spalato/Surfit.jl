using Surfit: resid_vs, ssq_of, scaled, unscaled, ssq_of, resid_vs, ssq_arr
using Surrogates
using Optim
using LinearAlgebra: norm
using Random

function surfit(model, x, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, initsamp)
    # setup scaled coordinates
    @assert all(lb .< guess .< ub) # check that guess is within bounds
    @assert all(lb .< ub) # check that bounds are valid
    @assert length(lb) == length(guess) == length(ub)
    spans = ub .- lb
    scale = scaled(identity, spans)
    unscale = unscaled(identity, spans)
    
    # compute true residuals and ssq for a given point
    true_resid = resid_vs(x, y_exp, model)
    true_ssq = ssq_of(true_resid)

    # define mapping to internal bounded coordinates, inspired by lmfit and MINUIT
    # see: https://lmfit.github.io/lmfit-py/bounds.html
    # https://github.com/lmfit/lmfit-py/blob/master/lmfit/parameter.py#L925 (setup_bounds)
    to_internal(p) = asin.(2*(p .- scale(lb)) ./ (scale(ub) .- scale(lb)) .- 1)
    from_internal(p) = scale(lb) .+ (scale(ub) .- scale(lb)) .* (0.5*(sin.(p) .+ 1)) # p_internal to p_bounded

    reg = 1e-32 # regularization term for the radial basis function
    sampler = SobolSample() # sampler for initial points # I just checked and: it returns always the same thing!

    
    print("Initializing")
    # Generate all corners of the hypercube defined by lb and ub
    corners = vec(collect(Iterators.product(zip(lb, ub)...)))
    sample_points = [tuple(scale(collect(corner))...) for corner in corners] # TODO: convert to vector of tuples
    # Fill the remaining sample points using sample(...)
    if length(sample_points) < initsamp
        # Generate Sobol sample points in the scaled space
        remaining_sample = sample(initsamp - length(sample_points), scale(lb), scale(ub), sampler)
        for s in remaining_sample
            push!(sample_points, s)
        end
    end

    push!(sample_points, tuple(scale(guess)...))
    print(" $(length(sample_points)) points")
    # compute initial values.
    samp_y = map(p -> model(x, unscale(p)...), sample_points)
    samp_ssq = map(y -> ssq_arr(y_exp, y), samp_y)
    @assert length(sample_points) == length(samp_y)
    @assert length(samp_y) == length(samp_ssq)
    # Outer loop variables
    current_min = scale(guess) # current minimum location
    min_index = length(sample_points) # current indexs
    current_ssq = true_ssq(current_min)

    while length(sample_points) < f_calls
        # Multi-output surrogate maps p -> y_guess
        surrogate = RadialBasis(sample_points, samp_y, scale(lb), scale(ub), rad=cubicRadial(); regularization=reg)
        #@assert surrogate(scale(guess)) ≈ model(x, guess...)
    
        bounded_surrogate = pi -> surrogate(from_internal(pi))
        # residuals of y_guess
        resid(p) = y_exp .- bounded_surrogate(p)
        # SSQ of y_exp and y_guess
        ssq = ssq_of(resid)
        #@assert ssq(scale(guess)) ≈ sum((y_exp - model(x, guess...)).^2)
        # inner loop: minimize the ssq using y_guess, bounded using MINUIT algorithm
        res = optimize(
            ssq, to_internal(current_min),
            NelderMead(),
            Optim.Options(
                store_trace=true,
                trace_simplex=true,
                allow_f_increases=false,

            );
        )
        new_min = from_internal(Optim.minimizer(res))
        y_true = model(x, unscale(new_min)...)
        new_ssq = ssq_arr(y_exp, y_true)
        push!(sample_points, tuple(new_min...))
        push!(samp_y, y_true)
        push!(samp_ssq, new_ssq)
        # output logging. Dirty.
        print("\rIt: $(length(sample_points)) surrogate fcalls $(Optim.f_calls(res)) $(round(new_ssq;digits=6)) $(round(current_ssq;digits=6)) at $(min_index)    ")
        

        # are we done?
        if (norm(unscale(new_min) .- unscale(current_min)) < x_tol) && (abs(new_ssq - current_ssq) < f_tol) # TODO: change to `isapprox`
            break
        else # there will be other conditions: out of bounds, small step
            nothing
        end
        min_index = argmin(samp_ssq)
        current_min = collect(sample_points[min_index])
        current_ssq = samp_ssq[min_index]
        
    end # while
    
end 

function proto()
    rng = Xoshiro(20250414)
    model(t, a, b) = a * exp.(-b * t) # Example model
    t = collect(0:0.1:10)                     # Example time data
    y_exp = model.(t, 2.0, 0.5) .+ 0.1 .* randn(rng, length(t)) # Add noise to experimental data
    guess = [1.8, 0.55]              # Initial guess
    lb = [0.5, 0.01]                 # Lower bounds
    ub = [3.0, 1.0]                  # Upper bounds
    x_tol = 1e-9                     # Tolerance for x
    f_tol = 1e-9                     # Tolerance for function value
    f_calls = 200                    # Maximum function calls
    initsamp = 25                    # Initial samples
    surfit(model, t, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, initsamp)
end
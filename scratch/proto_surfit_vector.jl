using Surfit: resid_vs, ssq_of, scaled, unscaled, ssq_of, resid_vs
using Surrogates
using Optim


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
    # compute initial values and build surrogate
    samp_val = map(p -> model(x, unscale(p)...), sample_points)
    current_min = scale(guess)
    min_index = length(sample_points)
    current_val = samp_val[min_index]

    # try a multi-output surrogate
    surrogate = RadialBasis(sample_points, samp_val, scale(lb), scale(ub), rad=cubicRadial(); regularization=reg)
    @assert surrogate(scale(guess)) ≈ model(x, guess...)
    
    resid(p) = y_exp .- surrogate(p)
    ssq = ssq_of(resid)
    @assert ssq(scale(guess)) ≈ sum((y_exp - model(x, guess...)).^2)
    res = optimize(
        ssq, scale(guess),
        NelderMead(),
        Optim.Options(
            store_trace=true,
            trace_simplex=true,
            allow_f_increases=false,

        );
    )
    # all seem ok until here.
    @infiltrate
end 

function proto()
    model(t, a, b) = a * exp.(-b * t) # Example model
    t = collect(0:0.1:10)                     # Example time data
    y_exp = model.(t, 2.0, 0.5) .+ 0.1 .* randn(length(t)) # Add noise to experimental data
    guess = [1.8, 0.55]              # Initial guess
    lb = [0.5, 0.01]                 # Lower bounds
    ub = [3.0, 1.0]                  # Upper bounds
    x_tol = 1e-6                     # Tolerance for x
    f_tol = 1e-6                     # Tolerance for function value
    f_calls = 200                    # Maximum function calls
    initsamp = 25                    # Initial samples
    surfit(model, t, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, initsamp)
end
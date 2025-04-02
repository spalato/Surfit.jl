#opt_compare.jl: compare NM optimize vs Surrogates for least-squares minimization
# Perform minimization using NM and RadialBasis Surrogate(s?).
# Try two models: single exp (3 parameters) and g1e1 (5 parameters)
# Compare the number of function evaluations.
# Store results... in DataFrame?
using DelimitedFiles
using Surrogates
using Optim
using Plots
using DataFrames
using CSV
using LinearAlgebra # Added to define `norm`
#using Radials: RadialBasis, thinplateRadial # Import updated RadialBasis

plotlyjs(size=(800, 600))

e1(t, a0, k0, along) = a0*exp(-k0*t) + along
g1e1(t, a0, k0, a1, k1, along) = a0*exp(-0.5*(k0*t)^2) + a1*exp(-k1*t) + along 

resid_vs(x, y, model) = p -> y .- model.(x, p...) # returns f(p::Array)::Array
ssq_of(resid) = p -> sum(resid(p).^2) # returns f(p::Array)::float
scaled(f, scales) = p -> f(p./scales)
unscaled(f, scales) = p -> f(p.*scales)

Base.zero(v::Tuple{Float64, Float64}) = (Base.zero(Float64), Base.zero(Float64))
Base.zero(::NTuple{N, T}) where {N, T} = ntuple(_ -> zero(T), N)

function default_optim(model, t, y_exp, guess)
    resid = resid_vs(t, y_exp, model)
    ssq = ssq_of(resid)

    # Nelder-Mead optimization
    res = optimize(
        ssq, guess,
        Optim.Options(store_trace=true, trace_simplex=true, allow_f_increases=false);
        autodiff = :forward
    )
    return (
        method="Nelder-Mead", model=string(model), popt=Optim.minimizer(res),
        vmin=Optim.minimum(res), fcalls=Optim.f_calls(res)
    )
end

function bounded_optim(model, t, y_exp, guess, lb, ub)
    resid = resid_vs(t, y_exp, model)
    ssq = ssq_of(resid)

    # Fminbox optimization with bounds
    res = optimize(
        ssq, lb, ub, guess,
        Fminbox(),
        Optim.Options(store_trace=true, trace_simplex=true, allow_f_increases=false);
        autodiff = :forward
    )
    return (
        method="LBFGS bounded", model=string(model), popt=Optim.minimizer(res),
        vmin=Optim.minimum(res), fcalls=Optim.f_calls(res)+Optim.g_calls(res)
    )
end

function surrogate_optim(model, t, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, initsamp=50)
    @info "Surrogate optimization"
    @info "model $(model) guess $(guess) lb $(lb) ub $(ub)"
    resid = resid_vs(t, y_exp, model)
    ssq = ssq_of(resid)
    @assert all(lb .< guess .< ub)
    @assert length(lb) == length(guess) == length(ub)
    spans = ub .- lb
    scale = scaled(identity, spans)
    unscale = unscaled(identity, spans)
    min_target = unscaled(ssq, spans)

    @assert unscale(scale(guess)) == guess

    # Generate all corners of the hypercube defined by lb and ub
    corners = vec(collect(Iterators.product(zip(lb, ub)...)))
    scaled_corners = [scale(collect(corner)) for corner in corners]
    @info "N corners: $(length(scaled_corners)) adding $(initsamp - length(scaled_corners))"
    # Fill the remaining sample points using sample(...)
    if length(scaled_corners) < initsamp
        # Generate Sobol sample points in the scaled space
        remaining_sample = sample(initsamp - length(scaled_corners), scale(lb), scale(ub), SobolSample())
    else
        # If we have enough corners, just use them
        remaining_sample = []
    end

    samp = vcat(scaled_corners, remaining_sample)
    push!(samp, tuple(scale(guess)...))
    samp_val = min_target.(samp)

    surrogate = RadialBasis(
        samp, samp_val, scale(lb), scale(ub), rad=cubicRadial();
        regularization=1e-12 # Add small regularization term
    )
    #@info "impact of regularization" surrogate(scale(guess)), ssq(guess)
    @assert isapprox(surrogate(scale(guess)), ssq(guess), rtol=1E-6)

    current_min = scale(guess)
    current_val = surrogate(current_min)
    
    # define mapping to internal bounded coordinates, inspired by lmfit and MINUIT
    # see: https://lmfit.github.io/lmfit-py/bounds.html
    # https://github.com/lmfit/lmfit-py/blob/master/lmfit/parameter.py#L925 (setup_bounds)
    # TODO: we are both scaling to [0,1] and using arcsin. This is not necessary.
    to_internal(p) = asin.(2*(p .- scale(lb)) ./ (scale(ub) .- scale(lb)) .- 1)
    from_internal(p) = scale(lb) .+ (scale(ub) .- scale(lb)) .* (0.5*(sin.(p) .+ 1)) # p_internal to p_bounded

    while length(samp) < f_calls
        # Step 1: Minimize surrogate using Nelder-Mead
        bounded_surrogate = pi -> surrogate(from_internal(pi))
        res = optimize(
            bounded_surrogate,  to_internal(current_min),
            NelderMead(),
            Optim.Options(
                store_trace=true,
                trace_simplex=true,
                outer_f_abstol=f_tol,
                f_abstol=f_tol,
                allow_f_increases=false,

            );
            autodiff = :forward
        )
        new_min = from_internal(Optim.minimizer(res))
        new_val = Optim.minimum(res)
        
        # Try: if we made a small step, add a small region around it. Like the latest simplex. Or the last centroid and its reflection through the point.
        if any(new_min .< scale(lb)) || any(new_min .> scale(ub))
            @warn "New minimum is out of bounds!"
            # add sample points using Sobol
            # Generate Sobol sample points in the scaled space
            new_samples = sample(10, scale(lb), scale(ub), SobolSample())
            # Add the new samples to the existing sample points
            samp = vcat(samp, new_samples)
            samp_val = vcat(samp_val, min_target.(new_samples))
        else
            # TODO: handle "small steps" more efficiently. Check last simplex and add it to the sample
            # Step 2: Evaluate target function at new minimum and update surrogate
            true_val = ssq(unscale(collect(new_min)))
            # if new_val > current_val
            #     @warn "New minimum is worse than current minimum!"
            # end
            # if true_val > current_val
            #     @warn "New minimum (true) is worse than current minimum!"
            # end
            push!(samp, tuple(new_min...))
            push!(samp_val, true_val)
            @info "IT $(length(samp)) surrogate NM fcalls $(Optim.f_calls(res)) $(new_val) $(true_val) $(current_val)"
        
            # Step 3: Check tolerances
            if norm(new_min .- current_min) < x_tol && abs(true_val - current_val) < f_tol
                return (
                    method="Surrogate", model=string(model), popt=unscale(collect(new_min)),
                    vmin=true_val, fcalls=length(samp)
                ), samp, samp_val
            end
        end
        



        min_index = argmin(samp_val)
        current_min = collect(samp[min_index])
        current_val = samp_val[min_index]
        surrogate = RadialBasis(
            samp, samp_val, scale(lb), scale(ub), rad=cubicRadial();
            regularization=1e-12 # Add small regularization term
        )
    end

    return (
        method="Surrogate", model=string(model), popt=unscale(collect(current_min)),
        vmin=current_val, fcalls=length(samp)
    ), samp, samp_val
end

function main()
    # load data
    @info "Loading data"
    fname = "data/Zsac_FLUPS_spectra_params.txt"
    data = readdlm(fname; comments=true)
    t = data[:,1]
    y_exp = data[:,3]
    # keep only between 0 and 2
    m = @. 0 < t < 2
    t = t[m]
    y_exp = y_exp[m]

    # Initial guesses and bounds
    guess_e1 = [0.5, 1/0.6, 2.5]
    lb_e1 = [0.4, 1/0.7, 2.45]
    ub_e1 = [0.6, 1/0.1, 2.58]

    guess_g1e1 = [0.25, 1/0.4, 0.25, 1/0.6, 2.5]
    lb_g1e1 = [0.1, 1/0.7, 0.1, 1/0.7, 2.45]
    ub_g1e1 = [0.6, 1/0.1, 0.6, 1/0.1, 2.58]

    # Benchmarks
    benchs = []

    # Perform optimization using Nelder-Mead
    ret_nm = default_optim(e1, t, y_exp, guess_e1)
    push!(benchs, ret_nm)
    @assert all(lb_e1 .< ret_nm[:popt] .< ub_e1)
    ret_nm = default_optim(g1e1, t, y_exp, guess_g1e1)
    push!(benchs, ret_nm)
    @assert all(lb_g1e1 .< ret_nm[:popt] .< ub_g1e1)

    # Perform optimization using bounded Fminbox
    push!(benchs, bounded_optim(e1, t, y_exp, guess_e1, lb_e1, ub_e1))
    push!(benchs, bounded_optim(g1e1, t, y_exp, guess_g1e1, lb_g1e1, ub_g1e1))

    # Perform optimization using surrogate
    ret = surrogate_optim(e1, t, y_exp, guess_e1, lb_e1, ub_e1, 1e-9, 1e-9, 200, 20)
    push!(benchs, ret[1])
    ret = surrogate_optim(g1e1, t, y_exp, guess_g1e1, lb_g1e1, ub_g1e1, 1e-6, 1e-6, 500)
    push!(benchs, ret[1])


    # Convert results to DataFrame
    df = DataFrame(benchs)
    # Save results to CSV file
    CSV.write("results/optim_results.csv", df)
    df
end
using DelimitedFiles
using Surrogates
using Optim
using LinearAlgebra # Added to define `norm`
using Infiltrator

resid_vs(x, y, model) = p -> y .- model.(x, p...) # returns f(p::Array)::Array
ssq_of(resid) = p -> sum(resid(p).^2) # returns f(p::Array)::float
scaled(f, scales) = p -> f(p./scales)
unscaled(f, scales) = p -> f(p.*scales)

Base.zero(v::Tuple{Float64, Float64}) = (Base.zero(Float64), Base.zero(Float64))
Base.zero(::NTuple{N, T}) where {N, T} = ntuple(_ -> zero(T), N)

function surrogatefit(model, t, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, initsamp=50)
    #@info "Surrogate optimization $(model)"
    #@info "model $(model) guess $(guess) lb $(lb) ub $(ub)"

    # create helper functions: residuals and sum of squares
    resid = resid_vs(t, y_exp, model)
    ssq = ssq_of(resid)

    # setup scaled coordinates
    @assert all(lb .< guess .< ub) # check that guess is within bounds
    @assert all(lb .< ub) # check that bounds are valid
    @assert length(lb) == length(guess) == length(ub)
    spans = ub .- lb
    scale = scaled(identity, spans)
    unscale = unscaled(identity, spans)
    min_target = unscaled(ssq, spans)
    #@assert unscale(scale(guess)) ≈ guess

    # define mapping to internal bounded coordinates, inspired by lmfit and MINUIT
    # see: https://lmfit.github.io/lmfit-py/bounds.html
    # https://github.com/lmfit/lmfit-py/blob/master/lmfit/parameter.py#L925 (setup_bounds)
    to_internal(p) = asin.(2*(p .- scale(lb)) ./ (scale(ub) .- scale(lb)) .- 1)
    from_internal(p) = scale(lb) .+ (scale(ub) .- scale(lb)) .* (0.5*(sin.(p) .+ 1)) # p_internal to p_bounded

    # To help explore around the minimum, we sometimes want to add a small
    # region around the new minimum. This is done by picking a simplex from the
    # trace of the inner loop and recentering it around the new minimum. We take
    # smaller and smaller simplexes. The following parameters define which simplex
    # to take and how this changes with iteration number.
    min_traj_scale = 0.25
    smplx_traj_scale = min_traj_scale
    traj_scale_step = 0.25

    reg = 1e-15 # regularization term for the radial basis function

    sampler = SobolSample() # sampler for initial points

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
    samp_val = min_target.(sample_points) 
    current_min = scale(guess)
    min_index = length(sample_points)
    current_val = samp_val[min_index]

    while length(sample_points) < f_calls
        @assert length(sample_points) == length(samp_val)
        # Inner loop: create surrogate and perform NM minimization on it.
        surrogate = RadialBasis(
            sample_points, samp_val, scale(lb), scale(ub), rad=cubicRadial();
            regularization=reg # Add small regularization term
        )
        bounded_surrogate = pi -> surrogate(from_internal(pi))
        res = optimize(
            bounded_surrogate, to_internal(current_min),
            NelderMead(),
            Optim.Options(
                store_trace=true,
                trace_simplex=true,
                allow_f_increases=false,

            );
        )
        new_min = from_internal(Optim.minimizer(res))
        true_val = ssq(unscale(collect(new_min)))
        push!(sample_points, tuple(new_min...))
        push!(samp_val, true_val)
        # TODO: check if we can reuse the last simplex as an input to our new one.
        # output logging. Dirty.
        print("\rIt: $(length(sample_points)) surrogate fcalls $(Optim.f_calls(res)) $(round(true_val;digits=6)) $(round(current_val;digits=6)) at $(min_index)    ")
        
        # If we got out of bounds, add samples
        # Currently we are not getting here, thanks to the bounded minimization.
        # We chould nevertheless check if NM wanted to get out of bounds. 
        if any(new_min .< scale(lb)) || any(new_min .> scale(ub))
            @warn "New minimum is out of bounds!"
            # add sample points
            new_samples = sample(10, scale(lb), scale(ub), RandomSample())
            # Add the new samples to the existing sample points
            sample_points = vcat(sample_points, new_samples)
            samp_val = vcat(samp_val, min_target.(new_samples))
        # Are we done?
        elseif (norm(new_min .- current_min) < x_tol) && (abs(true_val - current_val) < f_tol) # TODO: change to `isapprox`
            break
        # If step is small, add a simplex to the sample.
        # "small" is defined here as less than 1% of the distance between the bounds.
        elseif max(abs.(new_min .- current_min)...) < 0.01
            if smplx_traj_scale < 0.99
                # pick the simplex some fraction of the trace in.
                idx = Int(round(size(Optim.simplex_trace(res))[1] * smplx_traj_scale))
                simplex = from_internal.(Optim.simplex_trace(res)[idx])
                # recenter the simplex around the new minimumm add to the sample
                simplex = [new_min .+ (s .- new_min) for s in simplex]
                values = min_target.(simplex)
                sample_points = vcat(sample_points, simplex)
                samp_val = vcat(samp_val, values)

                # Pick the simplex later on next iteration.
                smplx_traj_scale = min(smplx_traj_scale+traj_scale_step,1)
            end
        # Step is large, we just add the new minimum, which is done already.
        # We also expand the simplex.
        else
            smplx_traj_scale = max(smplx_traj_scale-traj_scale_step, min_traj_scale)
        end
        min_index = argmin(samp_val)
        current_min = collect(sample_points[min_index])
        current_val = samp_val[min_index]

    end
    print("  Done\n")

    min_index = argmin(samp_val)
    current_min = collect(sample_points[min_index])
    current_val = samp_val[min_index]
    return unscale(collect(current_min)), current_val, unscale.(sample_points), samp_val
end
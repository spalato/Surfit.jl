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
using MAT
using SurrogatesPolyChaos
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
    @info "NM optimization $(model)"
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
    @info "LBFGS bounded optimization $(model)"
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
        vmin=Optim.minimum(res), fcalls=Optim.f_calls(res)+length(guess)*Optim.g_calls(res)
    )
end

# function make_surrogate(rad::Surrogates.RadialFunction, samp, samp_val, lb, ub)
#     return RadialBasis(
#         samp, samp_val, lb, ub, rad=rad;
#         regularization=1e-12 # Add small regularization term
#     )
# end

# function make_surrogate(sur::Kriging, samp, samp_val, lb, ub)
#     return Kriging(
#         samp, samp_val, lb, ub; p=fill(2, length(lb)), 
#     )
# end

function pure_surrogate(model, t, y_exp, guess, lb, ub, name, maker, optim, f_calls, initsamp=50)
    @info "Pure surrogate optimization $(model) $(name) $(string(optim))"
    #@info "model $(model) guess $(guess) lb $(lb) ub $(ub)"
    resid = resid_vs(t, y_exp, model)
    ssq = ssq_of(resid)
    @assert all(lb .< guess .< ub)
    @assert length(lb) == length(guess) == length(ub)
    spans = ub .- lb
    scale = scaled(identity, spans)
    unscale = unscaled(identity, spans)
    min_target = unscaled(ssq, spans)

    samp = sample(initsamp, scale(lb), scale(ub), SobolSample())
    samp_val = min_target.(samp)

    surrogate = maker(samp, samp_val, scale(lb), scale(ub))
    sur_res = surrogate_optimize(min_target, optim, scale(lb), scale(ub), surrogate, SobolSample(), maxiters=f_calls)
    return (
        method="$(name) $(string(optim))", model=string(model), popt=unscale(collect(sur_res[1])),
        vmin=sur_res[2], fcalls=length(samp)
    ), unscale.(samp), samp_val
end



function surrogate_optim(model, t, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, initsamp=50)
    @info "Surrogate optimization $(model)"
    #@info "model $(model) guess $(guess) lb $(lb) ub $(ub)"
    resid = resid_vs(t, y_exp, model)
    ssq = ssq_of(resid)
    @assert all(lb .< guess .< ub)
    @assert length(lb) == length(guess) == length(ub)
    spans = ub .- lb
    scale = scaled(identity, spans)
    unscale = unscaled(identity, spans)
    min_target = unscaled(ssq, spans)

    # simplex index scale
    min_traj_scale = 0.25
    smplx_traj_scale = min_traj_scale
    traj_scale_step = 0.25


    sampler = SobolSample()
    @assert unscale(scale(guess)) == guess
    print("Initializing")
    # Generate all corners of the hypercube defined by lb and ub
    corners = vec(collect(Iterators.product(zip(lb, ub)...)))
    scaled_corners = [scale(collect(corner)) for corner in corners] # TODO: convert to vector of tuples
   # @info "N corners: $(length(scaled_corners)) adding $(initsamp - length(scaled_corners))"
    # Fill the remaining sample points using sample(...)
    if length(scaled_corners) < initsamp
        # Generate Sobol sample points in the scaled space
        remaining_sample = sample(initsamp - length(scaled_corners), scale(lb), scale(ub), sampler)
    else
        # If we have enough corners, just use them
        remaining_sample = []
    end

    samp = vcat(scaled_corners, remaining_sample)
    push!(samp, tuple(scale(guess)...))
    samp_val = min_target.(samp)
    reg = 1e-15
    surrogate = RadialBasis(
        samp, samp_val, scale(lb), scale(ub), rad=cubicRadial();
        regularization=reg # Add small regularization term
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
        @assert length(samp) == length(samp_val)
        min_index = argmin(samp_val)
        current_min = collect(samp[min_index])
        current_val = samp_val[min_index]
        surrogate = RadialBasis(
            samp, samp_val, scale(lb), scale(ub), rad=cubicRadial();
            regularization=reg # Add small regularization term
        )
        # Step 1: Minimize surrogate using Nelder-Mead
        bounded_surrogate = pi -> surrogate(from_internal(pi))
        res = optimize(
            bounded_surrogate,  to_internal(current_min),
            NelderMead(),
            Optim.Options(
                store_trace=true,
                trace_simplex=true,
                allow_f_increases=false,

            );
            #autodiff = :forward
        )
        new_min = from_internal(Optim.minimizer(res))
        true_val = ssq(unscale(collect(new_min)))
        push!(samp, tuple(new_min...))
        push!(samp_val, true_val)

        print("\rIt: $(length(samp)) surrogate fcalls $(Optim.f_calls(res)) $(round(true_val;digits=6)) $(round(current_val;digits=6)) at $(min_index)    ")
        # Try: if we made a small step, add a small region around it. Like the latest simplex. Or the last centroid and its reflection through the point.
        if any(new_min .< scale(lb)) || any(new_min .> scale(ub))
            @warn "New minimum is out of bounds!"
            # add sample points using Sobol
            # Generate Sobol sample points in the scaled space
            new_samples = sample(10, scale(lb), scale(ub), RandomSample())
            # Add the new samples to the existing sample points
            samp = vcat(samp, new_samples)
            samp_val = vcat(samp_val, min_target.(new_samples))
        # are we done?
        elseif (norm(new_min .- current_min) < x_tol) && (abs(true_val - current_val) < f_tol) # TODO: change to `isapprox`

            break
        # If step is small, add the last simplex to the sample
        elseif max(abs.(new_min .- current_min)...) < 0.01
            #@info "Step is small, adding a simplex, $smplx_traj_scale"
            if smplx_traj_scale < 0.99
                # pick the simplex 20% in
                idx = Int(round(size(Optim.simplex_trace(res))[1] * smplx_traj_scale))
                smplx_traj_scale = min(smplx_traj_scale+traj_scale_step,1)
                simplex = from_internal.(Optim.simplex_trace(res)[idx])
                # recenter the simplex around the new minimum
                simplex = [new_min .+ (s .- new_min) for s in simplex]
                values = min_target.(simplex)
                samp = vcat(samp, simplex)
                samp_val = vcat(samp_val, values)
            end
        # Step is large, add the new minimum to the sample
        else
            #@info "Big step, adding new minimum"
            smplx_traj_scale = max(smplx_traj_scale-traj_scale_step, min_traj_scale)
                # push!(samp, tuple(new_min...))
                # push!(samp_val, true_val)
        end

    end
    print("  Done\n")

    min_index = argmin(samp_val)
    current_min = collect(samp[min_index])
    current_val = samp_val[min_index]
    return (
        method="Surrogate", model=string(model), popt=unscale(collect(current_min)),
        vmin=current_val, fcalls=length(samp)
    ), unscale.(samp), samp_val
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
    ub_e1 = [0.6, 1/0.2, 2.58]
    @info "Bounds for e1" lb_e1, ub_e1

    guess_g1e1 = [0.25, 1/0.4, 0.25, 1/0.6, 2.5]
    lb_g1e1 = [0.1, 1.5, 0.0, 1/0.8, 2.45]
    ub_g1e1 = [0.6, 10.0, 0.6, 1/0.3, 2.58]
    @info "bounds for g1e1" lb_g1e1, ub_g1e1

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

    # Perform optimization using pure surrogate
    make_cubic_radial = (samp, samp_val, lb, ub) -> RadialBasis(samp, samp_val, lb, ub, rad=cubicRadial(); regularization=1e-12)
    make_kriging = (samp, samp_val, lb, ub) -> Kriging(samp, samp_val, lb, ub; p=fill(2, length(lb)))
    for (name, sur, minimizer) in [
        ("Cubic Radial", make_cubic_radial, DYCORS()),
        ("Cubic Radial", make_cubic_radial, SRBF()), # Better than DYCORS
        ("Kriging", make_kriging, EI()), # Fails to converge to the correct minimum
       # ("Kriging", make_kriging, DYCORS()), # works poorly
    ]
        try
            ret = pure_surrogate(e1, t, y_exp, guess_e1, lb_e1, ub_e1, name, sur, minimizer, 200, 20)
        catch e
            @warn "Error in pure surrogate e1: $(e)"
        else
            push!(benchs, ret[1])
        end
        try
            ret = pure_surrogate(g1e1, t, y_exp, guess_g1e1, lb_g1e1, ub_g1e1, name, sur, minimizer, 500)
        catch e
            @warn "Error in pure surrogate g1e1: $(e)"
        else
            push!(benchs, ret[1])
        end
    end
    # Perform optimization using surrogate
    ret = surrogate_optim(e1, t, y_exp, guess_e1, lb_e1, ub_e1, 1e-9, 1e-12, 200, 50)
    push!(benchs, ret[1])
    ret = surrogate_optim(g1e1, t, y_exp, guess_g1e1, lb_g1e1, ub_g1e1, 1e-9, 1e-12, 200, 100)
    push!(benchs, ret[1])


    # Convert results to DataFrame
    df = DataFrame(benchs)
    # Save results to CSV file
    CSV.write("results/optim_results.csv", df)
    df
end
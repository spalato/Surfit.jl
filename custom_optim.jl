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
        Optim.Options(store_trace=true, trace_simplex=true);
        autodiff = :forward
    )
    return (
        method="Nelder-Mead", model=string(model), popt=Optim.minimizer(res),
        vmin=Optim.minimum(res), fcalls=Optim.f_calls(res)
    )
end

function surrogate_optim(model, t, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, initsamp=21)
    resid = resid_vs(t, y_exp, model)
    ssq = ssq_of(resid)

    spans = ub .- lb
    scale = scaled(identity, spans)
    unscale = unscaled(identity, spans)
    min_target = unscaled(ssq, spans)

    @assert unscale(scale(guess)) == guess
    samp = sample(initsamp, scale(lb), scale(ub), SobolSample())
    push!(samp, tuple(scale(guess)...))
    init_val = min_target.(samp)

    surrogate = RadialBasis(
        samp, init_val, scale(lb), scale(ub), rad=thinplateRadial();
        regularization=1e-12 # Add small regularization term
    )
    @info "impact of regularization" surrogate(scale(guess)), ssq(guess)
    @assert isapprox(surrogate(scale(guess)), ssq(guess), rtol=1E-6)

    current_min = scale(guess)
    current_val = surrogate(current_min)
    evaluations = length(samp)
    
    for nit in 1:(f_calls-initsamp)
        # Step 1: Minimize surrogate using Nelder-Mead
        res = optimize(
            surrogate, current_min,
            Optim.Options(store_trace=true, trace_simplex=true);
            autodiff = :forward
        )
        new_min = Optim.minimizer(res)
        new_val = Optim.minimum(res)


        # Step 2: Evaluate target function at new minimum and update surrogate
        true_val = ssq(unscale(collect(new_min)))
        push!(samp, tuple(new_min...))
        push!(init_val, true_val)

        @info "IT $(nit) surrogate NM fcalls $(Optim.f_calls(res)) $(new_val) $(true_val) $(current_val)"
        # Step 3: Check tolerances
        if norm(new_min .- current_min) < x_tol && abs(true_val - current_val) < f_tol
            return (
                method="Surrogate", model=string(model), popt=unscale(collect(new_min)),
                vmin=true_val, fcalls=evaluations
            )
        end

        min_index = argmin(init_val)
        current_min = collect(samp[min_index])
        current_val = init_val[min_index]
        surrogate = RadialBasis(
            samp, init_val, scale(lb), scale(ub), rad=thinplateRadial();
            regularization=1e-12 # Add small regularization term
        )
    end

    return (
        method="Surrogate", model=string(model), popt=unscale(collect(current_min)),
        vmin=current_val, fcalls=evaluations
    )
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
    lb_g1e1 = [0.1, 1/0.7, 0.1, 1/0.7, 2.5]
    ub_g1e1 = [0.6, 1/0.1, 0.6, 1/0.1, 2.58]

    # Benchmarks
    benchs = []

    # Perform optimization using Nelder-Mead
    push!(benchs, default_optim(e1, t, y_exp, guess_e1))
    push!(benchs, default_optim(g1e1, t, y_exp, guess_g1e1))

    # Perform optimization using surrogate
    push!(benchs, surrogate_optim(e1, t, y_exp, guess_e1, lb_e1, ub_e1, 1e-6, 1e-6, 100))
    push!(benchs, surrogate_optim(g1e1, t, y_exp, guess_g1e1, lb_g1e1, ub_g1e1, 1e-6, 1e-6, 100))

    # Convert results to DataFrame
    df = DataFrame(benchs)
    # Save results to CSV file
    CSV.write("results/optim_results.csv", df)
    df
end
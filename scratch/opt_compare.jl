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

plotlyjs(size=(800, 600))

e1(t, a0, k0, along) = a0*exp(-k0*t) + along
g1e1(t, a0, k0, a1, k1, along) = a0*exp(-0.5*(k0*t)^2) + a1*exp(-k1*t) + along 

resid_vs(x, y, model) = p -> y .- model.(x, p...) # returns f(p::Array)::Array
ssq_of(resid) = p -> sum(resid(p).^2) # returns f(p::Array)::float
scaled(f, scales) = p -> f(p./scales)
unscaled(f, scales) = p -> f(p.*scales)

Base.zero(v::Tuple{Float64, Float64}) = (Base.zero(Float64), Base.zero(Float64))
Base.zero(::NTuple{N, T}) where {N, T} = ntuple(_ -> zero(T), N)

function optimize_model(model, t, y_exp, guess, method_name, lb, ub, surrogate_method=false)
    resid = resid_vs(t, y_exp, model)
    ssq = ssq_of(resid)

    if !surrogate_method
        # Nelder-Mead optimization
        res = optimize(
            ssq, guess,
            Optim.Options(store_trace=true, trace_simplex=true);
            autodiff = :forward
        )
        return (
            method=method_name, model=string(model), popt=Optim.minimizer(res),
            vmin=Optim.minimum(res), fcalls=Optim.f_calls(res)
        )
    else
        # Surrogate optimization
        spans = ub .- lb
        scale = scaled(identity, spans)
        unscale = unscaled(identity, spans)
        min_target = unscaled(ssq, spans)

        @assert unscale(scale(guess)) == guess
        samp = sample(11, scale(lb), scale(ub), SobolSample())
        push!(samp, tuple(scale(guess)...))
        init_val = min_target.(samp)

        surrogate = Kriging(samp, init_val, scale(lb), scale(ub))
        @assert isapprox(surrogate(scale(guess)), ssq(guess))

        sur_res = surrogate_optimize(min_target, EI(), scale(lb), scale(ub), surrogate, RandomSample(), num_new_samples=10, maxiters=500)
        return (
            method=method_name, model=string(model), popt=unscale(collect(sur_res[1])),
            vmin=sur_res[2], fcalls=length(samp)
        )
    end
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
    # plot data
    pdata = scatter(t, y_exp)

    # Initial guesses and bounds
    guess_e1 = [0.5, 1/0.6, 2.5]
    lb_e1 = [0.4, 1/0.7, 2.45]
    ub_e1 = [0.6, 1/0.1, 2.58]

    guess_g1e1 = [0.25, 1/0.4, 0.25, 1/0.6, 2.5]
    lb_g1e1 = [0.1, 1/0.7, 0.1, 1/0.7, 2.5]
    ub_g1e1 = [0.6, 1/0.1, 0.6, 1/0.1, 2.58]

    # Benchmarks
    benchs = []

    # Nelder-Mead optimization
    push!(benchs, optimize_model(e1, t, y_exp, guess_e1, "NM", lb_e1, ub_e1))
    push!(benchs, optimize_model(g1e1, t, y_exp, guess_g1e1, "NM", lb_g1e1, ub_g1e1))

    # Surrogate optimization
    push!(benchs, optimize_model(e1, t, y_exp, guess_e1, "Kriging EI", lb_e1, ub_e1, true))
    push!(benchs, optimize_model(g1e1, t, y_exp, guess_g1e1, "Kriging EI", lb_g1e1, ub_g1e1, true))

    # Convert results to DataFrame
    df = DataFrame(benchs)
    df
end
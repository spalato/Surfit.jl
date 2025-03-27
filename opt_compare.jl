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

    # Define NM targets
    resid_e1 = resid_vs(t, y_exp, e1)
    ssq_e1 = ssq_of(resid_e1)

    resid_g1e1 = resid_vs(t, y_exp, g1e1)
    ssq_g1e1 = ssq_of(resid_g1e1)

    # inital guesses
    guess_e1 = [0.5, 1/0.6, 2.5]
    guess_g1e1 = [0.25, 1/0.4, 0.25, 1/0.6, 2.5]
    # plot!(pdata, t, e1.(t, guess_e1...))
    # plot!(pdata, t, g1e1.(t, guess_g1e1...))
    # # Perform NM optimization
    # display(pdata)

    res_e1 = optimize(
        ssq_e1, guess_e1, 
        Optim.Options(store_trace=true, trace_simplex=true);
        autodiff = :forward
    )
    benchs = []
    push!(
        benchs,
        (
            method="NM", model="e1", popt=Optim.minimizer(res_e1),
            vmin=Optim.minimum(res_e1), fcalls=Optim.f_calls(res_e1)
        )
    )
    @info "NM e1" Optim.minimum(res_e1) Optim.f_calls(res_e1)
    res_g1e1 = optimize(
        ssq_g1e1, guess_g1e1, 
        Optim.Options(store_trace=true, trace_simplex=true);
        autodiff = :forward
    )
    push!(benchs,
    (method="NM", model="g1e1",
    popt=Optim.minimizer(res_g1e1), vmin=Optim.minimum(res_g1e1),
    fcalls=Optim.f_calls(res_g1e1))
)
    @info "NM g1e1" Optim.minimum(res_g1e1) Optim.f_calls(res_g1e1)


    # surrogate fitting, e1
    lb = [0.4, 1/0.7, 2.45]
    ub = [0.6, 1/0.1, 2.58]
    spans = ub .- lb
    scale = scaled(identity, spans)
    unscale = unscaled(identity, spans)
    min_target = unscaled(ssq_e1, spans)

    @assert unscale(scale(guess_e1)) == guess_e1
    samp_e1 = sample(11, scale(lb), scale(ub), SobolSample())
    push!(samp_e1, tuple(scale(guess_e1)...))
    init_val = min_target.(samp_e1)

    surrogate = Kriging(samp_e1, init_val, scale(lb), scale(ub))
    @assert isapprox(surrogate(scale(guess_e1)), ssq_e1(guess_e1))

    sur_res_e1 = surrogate_optimize(min_target, EI(), scale(lb), scale(ub), surrogate, RandomSample(), num_new_samples=10, maxiters=500)
    push!(
        benchs,
        (
            method="Kriging EI", model="e1", popt=unscale(collect(sur_res_e1[1])),
            vmin=sur_res_e1[2], fcalls=length(samp_e1)
        )
    )
    # surrogate fitting, g1e1
    lb = [0.1, 1/0.7, 0.1, 1/0.7, 2.5]
    ub = [0.6, 1/0.1, 0.6, 1/0.1, 2.58]
    spans = ub .- lb
    scale = scaled(identity, spans)
    unscale = unscaled(identity, spans)
    min_target = unscaled(ssq_g1e1, spans)

    @assert unscale(scale(guess_g1e1)) == guess_g1e1
    samp_g1e1 = sample(11, scale(lb), scale(ub), SobolSample())
    push!(samp_g1e1, tuple(scale(guess_g1e1)...))
    init_val = min_target.(samp_g1e1)

    surrogate_g1e1 = Kriging(samp_g1e1, init_val, scale(lb), scale(ub))
    @assert isapprox(surrogate_g1e1(scale(guess_g1e1)), ssq_g1e1(guess_g1e1))

    sur_res_g1e1 = surrogate_optimize(min_target, EI(), scale(lb), scale(ub), surrogate_g1e1, RandomSample(), num_new_samples=10, maxiters=1000)
    push!(
        benchs,
        (
            method="Kriging EI", model="g1e1", popt=unscale(collect(sur_res_g1e1[1])),
            vmin=sur_res_g1e1[2], fcalls=length(samp_g1e1)
        )
    )


    df = DataFrame(benchs)
    #@infiltrate
    df
end
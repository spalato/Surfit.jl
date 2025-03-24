#opt_compare.jl: compare NM optimize vs Surrogates for least-squares minimization
# Perform minimization using NM and RadialBasis Surrogate(s?).
# Try two models: single exp (3 parameters) and g1e1 (5 parameters)
# Compare the number of function evaluations.
# Store results... in DataFrame?
using DelimitedFiles
using Surrogates
using Optim
using Plots

plotlyjs(size=(800, 600))

e1(t, a0, k0, along) = a0*exp(-k0*t) + along
g1e1(t, a0, k0, a1, k1, along) = a0*exp(-0.5*(k0*t)^2) + a1*exp(-k1*t) + along 

resid_vs(x, y, model) = p -> y .- model.(x, p...)
ssq_of(resid) = p -> sum(resid(p).^2)

Base.zero(v::Tuple{Float64, Float64}) = (Base.zero(Float64), Base.zero(Float64))

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
    plot!(pdata, t, e1.(t, guess_e1...))
    plot!(pdata, t, g1e1.(t, guess_g1e1...))
    # Perform NM optimization
    display(pdata)

    res_e1 = optimize(
        ssq_e1, guess_e1, 
        Optim.Options(store_trace=true, trace_simplex=true);
        autodiff = :forward
    )
    @info "NM e1" Optim.minimizer(res_e1) Optim.minimum(res_e1) Optim.f_calls(res_e1)
    res_g1e1 = optimize(
        ssq_g1e1, guess_g1e1, 
        Optim.Options(store_trace=true, trace_simplex=true);
        autodiff = :forward
    )
    @info "NM g1e1" Optim.minimizer(res_g1e1) Optim.minimum(res_g1e1) Optim.f_calls(res_g1e1)

end
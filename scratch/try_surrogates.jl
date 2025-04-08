# try_surrogates.jl: first try at surrogates.
# The objective is: load/generate some target data.
# Fit it using a simple model (e1), which is just an explicit function.
using Plots
using Random
using Surrogates
using SurrogatesPolyChaos
using Optim
#https://modernjuliaworkflows.org/writing/#repl

plotlyjs(size=(800, 600))

e1(t, a0, k0) = t>0 ? a0*exp(-k0*t) : 0

Base.zero(v::Tuple{Float64, Float64}) = Base.zero(collect(v))

function deep_flatten(v)
    if v isa Vector
        return vcat(map(deep_flatten, v)...)  # Recursively flatten each element
    else
        return [v]  # Wrap non-vector elements in a list
    end
end

function gen_data()
    t = range(-10.0, 100.0, step=2.0)
    Random.seed!(20250320)
    noise = 0.1*randn((size(t)))
    a0 = 5.0
    k0 = 1/20.0
    y = e1.(t, a0, k0)
    t, y+noise, y
end

function main()
    # Generate target dataset single exp
    t, yexp, y = gen_data()
    # residual function
    resid(p) = yexp - e1.(t, p...)
    # sum of squares
    ssq(p) = sum(resid(p).^2)
    
    # initial guess
    guess = [2.0, 1/18.0]
    yg = e1.(t, guess...)    
    rg = resid(guess)
    @info "SSQ" ssq(guess)
    l = @layout [a{0.8h};b]
    pf = scatter(t, yexp, label="yexp")
    plot!(pf, t, y, label="true")
    plot!(pf, t, yg, label="guess")
    pr = scatter(t, rg)
    display(plot(pf, pr, layout=l, label="r"))
    
    # optimize using Optimization.jl
    res = optimize(ssq, guess, Optim.Options(store_trace=true, trace_simplex=true); autodiff = :forward)#, Optim.Options(trace_simplex=true))
    @info "Simplex loc value niter" Optim.minimizer(res) Optim.minimum(res) Optim.f_calls(res)
    simplex_trace = stack(stack(Optim.simplex_trace(res)))
    simplex_trace = reshape(simplex_trace, (2, size(simplex_trace)[2]*size(simplex_trace)[3]))

    # now onto surrogates
    # bounds for parameters
    lb = [0.0, 0.01]
    ub = [10.0, 0.1]
    scales = ub .- lb

    unscale(p) = p.*scales
    scale(x) = x./scales
    #@infiltrate
    lbs = scale(lb)
    ubs = scale(ub)
    guess_s = scale(guess)
    scaled_ssq(p) = ssq(unscale(p))
    nsamples = 25
    sampling = SobolSample()
    xsamp = sample(nsamples, lbs, ubs, SobolSample())
    ysamp = scaled_ssq.(xsamp)
    #@info "Initial sampling" xsamp ysamp
    # Radial (default): bad
    # Kriging (default): smooth
    # Lobachevsky : bad.
    # SecondOrderPolynomialSurrogate : very smooth!
    # Wendland Completely off.
    # Lobachevsky with alpha=1.0, 10 seems to do it. That would be 1/scale, more or less.
    #surrogate = Kriging(xsamp, ysamp, lb, ub, p=[1.5, 1.5], theta=[0.1, 1.0])#[1/(u-l) for (l, u) in zip(lb, ub)]) 
    #surrogate = Kriging(xsamp, ysamp, lbs, ubs, p=[1.5, 1.5], theta=[10.0, 10.0])
    # Lobachevsky seems to crash on a linear algebra problem (like adding the same point many times).
    #surrogate = LobachevskySurrogate(xsamp, ysamp, lbs, ubs, alpha=[0.01, 0.01]) 
    #surrogate = PolynomialChaosSurrogate(xsamp, ysamp, lb, ub)
    #@infiltrate
    surrogate = RadialBasis(xsamp, ysamp, lbs, ubs)
    @info "Estimation, at guess" surrogate(guess_s) ssq(guess)
    
    # compute the rms distance between the true result and the surrogate over initial samples
    #@info "Initial RMS error" sum(@. (surrogate(xsamp) - ysamp)^2)
    # plot the true surface, the initial surrogate and the final surrogate.
    x_smooth = range(lb[1], ub[1], 256)
    y_smooth = range(lb[2], ub[2], 256)
    levels = 0:5:50
    #@infiltrate
    #surf_true = contourf(x_smooth, y_smooth, (x, y)->ssq([x y]), levels=levels)
    surf_scaled = contourf(x_smooth./scales[1], y_smooth./scales[2], (x, y)->scaled_ssq((x, y)), levels=levels)

    surf_init = contourf(x_smooth/scales[1], y_smooth/scales[2], (x, y)->surrogate([x y]), levels=levels)
    p1s = [xy[1] for xy in xsamp]
    p2s = [xy[2] for xy in xsamp]
    scatter!(surf_init, p1s, p2s,  markercolor=:white, cbar=false)
    
 #   # optimizing
    sur_res = surrogate_optimize(scaled_ssq, DYCORS(), lbs, ubs, surrogate, sampling,  num_new_samples = 200)
    @info "Surrogate optimize complete" unscale(sur_res[1]) sur_res[2]
    surf_end = contourf(x_smooth/scales[1], y_smooth/scales[2], (x, y)->surrogate([x y]), levels=levels)
    p1s = [xy[1] for xy in xsamp]
    p2s = [xy[2] for xy in xsamp]
    scatter!(surf_end, p1s, p2s, markercolor=:white, cbar=false)
    surf_err = contourf(x_smooth/scales[1], y_smooth/scales[2], 
    (x, y)->scaled_ssq((x, y))-surrogate([x y]), levels=-4:0.5:4, color=:RdBu)
    surfs = [surf_scaled, surf_init, surf_end,surf_err]
    
    for s in surfs
        scatter!(s, scale(simplex_trace)[1,:], scale(simplex_trace)[2,:], markercolor=:gray)
        scatter!(s, [guess_s[1]], [guess_s[2]], markercolor=:red)
        scatter!(s, [5.0/scales[1]], [1/20.0/scales[2]], markercolor=:blue)
        scatter!(s, [sur_res[1][1]], [sur_res[1][2]], markercolor=:green)
        
    end

    display(plot(surfs..., layout=(2,2), size=(600, 600), legend=false))
    @info "Sample length from .. to " nsamples length(xsamp)
end

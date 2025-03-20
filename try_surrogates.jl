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
    res = optimize(ssq, guess; autodiff = :forward)
    @info "Simplex loc value niter" Optim.minimizer(res) Optim.minimum(res) Optim.iterations(res)
    #@infiltrate

    # now onto surrogates
    # bounds for parameters
    lb = [0.0, 1/100.0]
    ub = [10.0, 1/2.0]
    nsamples = 25
    sampling = HaltonSample()
    xsamp = sample(nsamples, lb, ub,GridSample())
    ysamp = ssq.(xsamp)
    #@info "Initial sampling" xsamp ysamp
    # Radial (default): bad
    # Kriging (default): smooth
    # Lobachevsky : bad.
    # SecondOrderPolynomialSurrogate : very smooth!
    # Wendland Completely off.
    # Lobachevsky with alpha=1.0, 10 seems to do it. That would be 1/scale, more or less.
    #surrogate = Kriging(xsamp, ysamp, lb, ub, p=[1.5, 1.5], theta=[0.1, 1.0])#[1/(u-l) for (l, u) in zip(lb, ub)]) 
    surrogate = Kriging(xsamp, ysamp, lb, ub, p=[2.0, 2.0], theta=[10.0/(u-l) for (l, u) in zip(lb, ub)]) 
    # Lobachevsky seems to crash on a linear algebra problem (like adding the same point many times).
    #surrogate = LobachevskySurrogate(xsamp, ysamp, lb, ub, alpha=[0.01, 0.01]) 
    #surrogate = PolynomialChaosSurrogate(xsamp, ysamp, lb, ub)
    @info "Estimation, at guess" surrogate(guess) ssq(guess)
    # plot the surrogate
    x_smooth = range(lb[1], ub[1], 64)
    y_smooth = range(lb[2], ub[2], 64)

    surface(x_smooth, y_smooth, (x, y)->surrogate([x y]))
    p1s = [xy[1] for xy in xsamp]
    p2s = [xy[2] for xy in xsamp]
    display(scatter!(p1s, p2s, ysamp, marker_z = ysamp, markercolor=:black, cbar=false))
 #   # optimizing
    sur_res = surrogate_optimize(ssq, EI(), lb, ub, surrogate, sampling)
    @info "Surrogate optimize complete" sur_res[1] sur_res[2]
    surface(x_smooth, y_smooth, (x, y)->surrogate([x y]))
    p1s = [xy[1] for xy in xsamp]
    p2s = [xy[2] for xy in xsamp]
    display(scatter!(p1s, p2s, ysamp, marker_z = ysamp, markercolor=:black, cbar=false))
    @info "Sample length from .. to " nsamples length(xsamp)
end

# try_surrogates.jl: first try at surrogates.
# The objective is: load/generate some target data.
# Fit it using a simple model (e1), which is just an explicit function.
using Plots
using Random
using Surrogates
#https://modernjuliaworkflows.org/writing/#repl



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
    guess = (5.0, 1/20.0)
    yg = e1.(t, guess...)    
    rg = resid(guess)
    @info "SSQ" ssq(guess)
    l = @layout [a{0.8h};b]
    pf = scatter(t, yexp, label="yexp")
    plot!(pf, t, y, label="true")
    plot!(pf, t, yg, label="guess")
    pr = scatter(t, rg)
    plot(pf, pr, layout=l, label="r")
    
end

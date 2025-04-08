using Plots
default(c = :matter, legend = false, xlabel = "x", ylabel = "y")
using Surrogates

Base.zero(v::Tuple{Float64, Float64}) = (zero(Float64), zero(Float64))

function booth(x)
    x1 = x[1]
    x2 = x[2]
    term1 = (x1 + 2 * x2 - 7)^2
    term2 = (2 * x1 + x2 - 5)^2
    y = term1 + term2
end

function main()
    n_samples = 100
    lower_bound = [-5.0, 0.0]
    upper_bound = [10.0, 15.0]

    xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
    zs = booth.(xys)
    x, y = -5.0:10.0, 0.0:15.0
    p1 = surface(x, y, (x1, x2) -> booth((x1, x2)))
    xs = [xy[1] for xy in xys]
    ys = [xy[2] for xy in xys]
    scatter!(xs, ys, zs)
    p2 = contour(x, y, (x1, x2) -> booth((x1, x2)))
    scatter!(xs, ys)
    plot(p1, p2, title = "True function")

    radial_basis = RadialBasis(xys, zs, lower_bound, upper_bound, rad=thinplateRadial())

    p1 = surface(x, y, (x, y) -> radial_basis([x y]))
    scatter!(xs, ys, zs, marker_z = zs)
    p2 = contour(x, y, (x, y) -> radial_basis([x y]))
    scatter!(xs, ys, marker_z = zs)
    plot(p1, p2, title = "Surrogate")
    @info "initial size" size(xys)

    
    res = surrogate_optimize(
        booth, DYCORS(), lower_bound, upper_bound, radial_basis, RandomSample(), maxiters = 50)
    @info "final size" size(xys)

    res
end
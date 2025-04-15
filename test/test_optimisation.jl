using Test
using Surrogates: sample, SobolSample
#using Random

# Import the functions to be tested
include("../src/optimisation.jl")

# Mock model function for testing
function mock_model(x, p...)
    return p[1] .* x .+ p[2]
end

# Test for `resid_vs`
@testset "resid_vs tests" begin
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    model = mock_model
    p = [2.0, 0.0]
    resid = resid_vs(x, y, model)
    @test resid(p) ≈ [0.0, 0.0, 0.0]
end

# Test for `ssq_of`
@testset "ssq_of tests" begin
    resid = p -> [1.0, 2.0, 3.0] .- p
    ssq = ssq_of(resid)
    @test ssq([1.0, 2.0, 3.0]) ≈ 0.0
    @test ssq([2.0, 2.0, 3.0]) ≈ 1.0
    @test ssq([2.0, 3.0, 3.0]) ≈ 2.0
    @test ssq([1.0, 2.0, 1.0]) ≈ 4.0
end

# Test for `scaled` and `unscaled`
@testset "scaled and unscaled identity" begin
    scales = [2.0, 3.0]
    scale = scaled(identity, scales)
    unscale = unscaled(identity, scales)
    params = ([5.0, 10.0],)
    for p in params
        @test unscale(scale(p)) == p
    end
end

# Test for `scaled` and `unscaled`
@testset "scaled and unscaled tests" begin
    f = p -> sum(p)
    scales = [2.0, 3.0]
    scaled_f = scaled(f, scales)
    unscaled_f = unscaled(f, scales)
    scale = scaled(identity, scales)
    unscale = unscaled(identity, scales)
    params  = ([4.0, 6.0],)
    for p in params
        @test scaled_f(p) == f(scale(p))
    end
end

# Test for `resid_arr` and `ssq_arr`
@testset "resid_arr and ssq_arr tests" begin
    y_exp = [1.0, 2.0, 3.0]
    y = [1.5, 2.5, 3.5]
    @test resid_arr(y_exp, y) ≈ [-0.5, -0.5, -0.5]
    @test ssq_arr(y_exp, y) ≈ 0.75
end

# Test for `surfit_scalar`
@testset "surfit_scalar tests" begin
    ssq = p -> sum((p .- [1.0, 2.0]).^2)
    guess = [0.0, 0.0]
    lb = [-5.0, -5.0]
    ub = [5.0, 5.0]
    x_tol = 1e-6
    f_tol = 1e-6
    f_calls = 100
    result, value, sample_points, samp_val = surfit_scalar(ssq, guess, lb, ub, x_tol, f_tol, f_calls)
    @test isapprox(result, [1.0, 2.0], atol=1e-2)
    @test isapprox(value, 0.0, atol=1e-2)
end

# Test for `surfit`
@testset "surfit tests" begin
    x = [1.0, 2.0, 3.0]
    y_exp = [3.0, 5.0, 7.0]
    guess = [1.0, 1.0]
    lb = [0.0, 0.0]
    ub = [5.0, 5.0]
    x_tol = 1e-6
    f_tol = 1e-6
    f_calls = 100
    result, value, sample_points, samp_y = surfit(mock_model, x, y_exp, guess, lb, ub, x_tol, f_tol, f_calls)
    @test isapprox(result, [2.0, 1.0], atol=1e-2)
    @test isapprox(value, 0.0, atol=1e-2)
end

@testset "surfit tests array init" begin
    x = [1.0, 2.0, 3.0]
    y_exp = [3.0, 5.0, 7.0]
    guess = [1.0, 1.0]
    lb = [0.0, 0.0]
    ub = [5.0, 5.0]
    x_tol = 1e-6
    f_tol = 1e-6
    f_calls = 100
    corners = vec(collect(Iterators.product(zip(lb, ub)...)))
    sample_points = [tuple(collect(corner)...) for corner in corners]
    extra = sample(20, lb, ub, SobolSample())
    for p in extra
        push!(sample_points, p)
    end
    result, value, sample_points, samp_y = surfit(mock_model, x, y_exp, guess, lb, ub, x_tol, f_tol, f_calls, sample_points)
    @test isapprox(result, [2.0, 1.0], atol=1e-2)
    @test isapprox(value, 0.0, atol=1e-2)
end
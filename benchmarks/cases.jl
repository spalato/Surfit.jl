using DelimitedFiles

e1(x, a0, k0, along) = a0 .* exp.(-k0 .* x) .+ along
g1e1(x, a0, k0, a1, k1, along) =
    a0 .* exp.(-0.5 .* (k0 .* x).^2) .+ a1 .* exp.(-k1 .* x) .+ along

function benchmark_cases(data_path)
    data = readdlm(data_path; comments=true)
    x = data[:, 1]
    y = data[:, 3]
    selected = @. 0 < x < 2
    x = x[selected]
    y = y[selected]

    [
        (
            case=BenchmarkCase("e1", e1, x, y),
            scenarios=scenario_variants(
                "default",
                [0.5, 1 / 0.6, 2.5],
                [0.4, 1 / 0.7, 2.45],
                [0.6, 1 / 0.2, 2.58],
            ),
        ),
        (
            case=BenchmarkCase("g1e1", g1e1, x, y),
            scenarios=scenario_variants(
                "default",
                [0.25, 1 / 0.4, 0.25, 1 / 0.6, 2.5],
                [0.1, 1.5, 0.0, 1 / 0.8, 2.45],
                [0.6, 10.0, 0.6, 1 / 0.3, 2.58],
            ),
        ),
    ]
end
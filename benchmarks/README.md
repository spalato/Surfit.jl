# Surfit benchmarks

Run the benchmark from the repository root:

```powershell
julia +lts --project=. benchmarks/run_benchmark.jl
```

The command writes a CSV to `benchmarks/results/benchmark.csv`. Each model and
scenario produces one row for Levenberg-Marquardt, which is the reference, and
one row for each selected comparison algorithm. L-BFGS remains an independent
performance row.

Run a smaller filtered benchmark:

```powershell
julia +lts --project=. benchmarks/run_benchmark.jl `
    --models=e1 `
    --scenarios=default `
    --algorithms=L-BFGS,Nelder-Mead `
    --seed=123
```

Use the other installed Julia channel in the same way:

```powershell
julia +release --project=. benchmarks/run_benchmark.jl
```

The benchmark has been verified with both channels. For a quick cross-version
check, run the focused tests with:

```powershell
julia +lts --project=. -e 'include("test/test_benchmark.jl")'
julia +release --project=. -e 'include("test/test_benchmark.jl")'
```

Every output row also records the Julia and Surfit versions, seed, and run
timestamp. The benchmark data is stored at
`benchmarks/data/Zsac_FLUPS_spectra_params.txt`; use an explicit seed when
comparing runs. The timestamp is provenance metadata and is not expected to
match.
`model_calls` is the comparable evaluation metric. Solver-specific fields such
as `solver_f_calls` are diagnostics and are not combined with it. Levenberg-Marquardt uses LsqFit's
numerical finite-difference Jacobian path (`autodiff=:finiteforward` by
default), so `solver_f_calls` counts residual callback invocations and
`solver_g_calls` is left empty because LsqFit does not expose its internal
finite-difference column evaluations separately.

All algorithms use the same hardcoded `x_tol=1e-8` and `f_tol=1e-10` policy.
For LsqFit, `x_tol` maps to its parameter tolerance and `f_tol` maps to its
gradient stopping tolerance because LsqFit does not expose a function-value
tolerance keyword.
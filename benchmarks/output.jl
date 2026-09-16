using CSV
using DataFrames
using Dates
using Surfit

function benchmark_metadata(; seed)
    (
        julia_version=string(VERSION),
        surfit_version=string(Base.pkgversion(Surfit)),
        seed=seed,
        run_timestamp=string(Dates.now()),
    )
end

function add_metadata(rows, metadata)
    [merge(row, metadata) for row in rows]
end

function write_results(path, rows)
    mkpath(dirname(path))
    CSV.write(path, DataFrame(rows); transform=(column, value) -> something(value, missing))
    path
end
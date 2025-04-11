using Surfit: hash_call, stored, store!, get_result, get_param
using HDF5

cases = [
    ("f1", (1, 2, 3), "0x404c404"),
    ("f1", (2, 2, 3), "0xcf38f567"),
    ("f2", (4.5, 6.7), "0xb241c6f"),
    ("f2", (4.500000000000001, 6.7), "0xf9281191"),
    ("f3", (4.5, 6.7, 0.0), "0xee894544"),
]

for (name, params, value) in cases
    # testing against hard coded values to ensure it does not depend on julia
    # version or invocation
    @test hash_call(params) == value
end


# From: https://discourse.julialang.org/t/macro-for-counting-the-number-of-times-a-function-is-called/3129
mutable struct Counting{TF}
    f::TF
    counter::Int
end

Counting(f) = Counting(f, 0)
reset!(c::Counting) = c.counter = 0

function (c::Counting)(args...)
    c.counter += 1
    c.f(args...)
end

# Define test functions
f1(x, y) = x + y
f2(x, y, z) = x * y * z
f3(x) = x^2

@testset "stored function tests" begin
    # Create a temporary HDF5 file
    temp_h5_file = tempname() * ".h5"
    h5file = h5open(temp_h5_file, "w")

    cases = Dict(
        f1 => [(1,2),(1.0, 2.0)],  # mixed types crash the thing. We will be using float arrays anyway.
        f2 => [(2.0,3.0,4.0),],
        f3 => [(5.0,),],
    )
    try
        for (f, params) in cases
            # create a group to store the results.
            counted_f = Counting(f)
            group = create_group(h5file, string(f))
            stored_f = stored(counted_f, group)
            for p in params
                # Test cases
                r = f(p...)
                @test stored_f(p...) == r
                @test haskey(group, hash_call(p))  # Ensure result is cached
                @test get_result(group, hash_call(p)) == r
                @test get_param(group, hash_call(p)) == p
                
                count = counted_f.counter  # Grab current execution count
                @test stored_f(p...) == r
                @test counted_f.counter == count  # Ensure no recomputation
            end
        end
    finally
        close(h5file)
        rm(temp_h5_file, force=true)
    end
end

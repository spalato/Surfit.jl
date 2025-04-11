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
    # version or invokation
    @test hash_call(params) == value
end

# Define test functions with counters. This is dirty AI Slop.
# Better implementations: https://discourse.julialang.org/t/macro-for-counting-the-number-of-times-a-function-is-called/3129
f1_counter = Ref(0)
f2_counter = Ref(0)
f3_counter = Ref(0)

f1(x, y) = begin
    f1_counter[] += 1
    x + y
end

f2(x, y, z) = begin
    f2_counter[] += 1
    x * y * z
end

f3(x) = begin
    f3_counter[] += 1
    x^2
end

@testset "stored function tests" begin
    # Create a temporary HDF5 file
    temp_h5_file = tempname() * ".h5"
    h5file = h5open(temp_h5_file, "w")

    cases = Dict(
        f1 => (f1_counter, [(1,2),(1.0, 2.0)]),  # mixed types crash the thing. We will be using float arrays anyway.
        f2 => (f2_counter, [(2.0,3.0,4.0),]),
        f3 => (f3_counter, [(5.0,),]),
    )
    try
        for (f, (counter, params)) in cases
            # create a group to store the results.
            group = create_group(h5file, string(f))
            stored_f = stored(f, group)
            for p in params
                # Test cases
                r = f(p...)
                @test stored_f(p...) == r
                @test haskey(group, hash_call(p))  # Ensure result is cached
                @test get_result(group, hash_call(p)) == r
                @test get_param(group, hash_call(p)) == p

                counter[] = 0
                @test stored_f(p...) == r
                @test counter[] == 0  # Ensure no recomputation
            end
        end
    finally
        close(h5file)
        rm(temp_h5_file, force=true)
    end
end

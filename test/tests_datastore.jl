using Surfit: hash_call, stored, store!, get_result
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

# Define test functions with counters
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

    try
        # Create groups for each function
        group_f1 = create_group(h5file, "f1")
        group_f2 = create_group(h5file, "f2")
        group_f3 = create_group(h5file, "f3")

        # Wrap functions with `stored`
        stored_f1 = stored(f1, group_f1)
        stored_f2 = stored(f2, group_f2)
        stored_f3 = stored(f3, group_f3)

        # Test cases
        @test stored_f1(1, 2) == f1(1, 2)
        @test haskey(group_f1, hash_call((1, 2)))  # Ensure result is cached
        @test get_result(group_f1, hash_call((1, 2))) == f1(1, 2)

        @test stored_f2(2, 3, 4) == f2(2, 3, 4)
        @test haskey(group_f2, hash_call((2, 3, 4)))  # Ensure result is cached
        @test get_result(group_f2, hash_call((2, 3, 4))) == f2(2, 3, 4)

        @test stored_f3(5) == f3(5)
        @test haskey(group_f3, hash_call((5,)))  # Ensure result is cached
        @test get_result(group_f3, hash_call((5,))) == f3(5)

        # Ensure second calls retrieve stored values and do not recompute
        f1_val = f1(1, 2)
        f2_val = f2(2, 3, 4)
        f3_val = f3(5)
        f1_counter[] = 0
        f2_counter[] = 0
        f3_counter[] = 0

        @test stored_f1(1, 2) == f1_val
        @test f1_counter[] == 0  # Ensure no recomputation

        @test stored_f2(2, 3, 4) == f2_val
        @test f2_counter[] == 0  # Ensure no recomputation

        @test stored_f3(5) == f3_val
        @test f3_counter[] == 0  # Ensure no recomputation
    finally
        close(h5file)
        rm(temp_h5_file, force=true)
    end
end

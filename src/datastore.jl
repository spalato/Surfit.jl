# datastore.jl
# This module provides on-disk caching for target functions, thereby accelerating
# surrogate fitting. It is designed to be used with the Surfit.jl package.
using CRC32c
using HDF5

"""
    hash_call(f, p): Generate a unique hash from the function name and parameters

The hash uses CRC32c, and should remain constant accross versions.
"""
function hash_call(p)
    # hash the name strin
    # take!(io) returns a Vector{UInt8} of the bytes in `p`
    io = IOBuffer()
    write(io, Ref(p))
    # hash the content with the name
    "0x$(string(crc32c(take!(io)), base=16))"
end


# store the return array into "result", parameters into "param"
# We could be more flexible by storing results, parameters, and independant
# variable. This would have to be specified in "store".
# Not needed for now... (check!)
# HDF5DataStore is supertype of both file and group, therefore both can act as stores
get_result(ds::HDF5.H5DataStore, key) = read(ds[key], "result")
get_param(ds::HDF5.H5DataStore, key) = Tuple(read(ds[key], "param"))
get_indep(ds::HDF5.H5DataStore, key) = read(ds[key], "indep")

function store!(ds::HDF5.H5DataStore, key::AbstractString, parameters::NTuple{N, Any}, result) where {N}
    group = create_group(ds, key)
    try
        group["result"] = result
        group["param"] = collect(parameters) # we will run into issues with mixed integer and floats
    catch e
        # something failed. We delete the group
        delete_object(group)
        rethrow(e)
    end
    nothing
end

function store!(ds::HDF5.H5DataStore, key::AbstractString, parameters::NTuple{N, Any}, indeps, result) where {N}
    group = create_group(ds, key)
    try
        group["indep"] = indeps
        group["result"] = result
        group["param"] = collect(parameters) # we will run into issues with mixed integer and floats
    catch e
        # something failed. We delete the group
        delete_object(group)
        rethrow(e)
    end
    nothing
end


function stored_scalar(f, store::HDF5.H5DataStore)
    (p...) -> begin
        # compute hash
        key = hash_call(p)
        # check if hash is in store
        if haskey(store, key)
            # if so, return cached result
            @debug "Loading $(key)"
            return get_result(store, key)
        else
            # if not, calculate f(p), store result then return it.
            res = f(p...)
            @debug "Storing $(key)"
            store!(store, key, p, res)
            return res
        end
    end
end

function stored(f, store::HDF5.H5DataStore)
    (x, p...) -> begin
        # compute hash
        key = hash_call(p)
        # check if hash is in store
        if haskey(store, key)
            # if so, return cached result
            @debug "Loading $(key)"
            stored_x = get_indep(store, key)
            (size(stored_x) == size(x) &&(stored_x ≈ x)) || throw(ArgumentError("Stored independents are different."))
            return get_result(store, key)
        else
            # if not, calculate f(p), store result then return it.
            res = f.(x, p...)
            @debug "Storing $(key)"
            store!(store, key, p, x, res)
            return res
        end
    end
end
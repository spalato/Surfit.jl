using HDF5

# struct DataStore
#     fid::HDF5.File
# end

# DataStore(filename, mode="cw") = DataStore(h5open(filename, mode))

# # opening, closeing file
# Base.isopen(ds::DataStore) = Base.isopen(ds.fid)
# Base.close(ds::DataStore) = Base.close(ds.fid)

# # Finding and storing things
# Base.keys(ds::DataStore) = Base.keys(ds.fid)
# Base.isempty(ds::DataStore)::Bool = Base.isempty(ds.fid)

# # TODO: empty!() remove everything
# Base.length(ds::DataStore) = Base.length(ds.fid)
# Base.haskey(ds::DataStore, key)::Bool = Base.haskey(ds.fid, key)
# Base.getindex(ds::DataStore, key)::HDF5.Group = Base.haskey(ds.fid, key) # get the group
get_result(ds::HDF5.H5DataStore, key) = read(ds[key], "result")
get_param(ds::HDF5.H5DataStore, key) = Tuple(read(ds[key], "param"))

function store!(ds::HDF5.H5DataStore, key::AbstractString, parameters::NTuple{N, AbstractFloat}, result::AbstractArray) where {N}
    group = create_group(ds, key)
    try
        group["result"] = result
        group["param"] = collect(parameters)
    catch e
        # something failed. We delete the group
        delete_object(group)
        rethrow(e)
    end
    nothing
end
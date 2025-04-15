using DelimitedFiles
using Surrogates
using Surfit
using HDF5
using DataStructures

e1(t, a0::Float64, k0::Float64, along::Float64) = a0*exp(-k0*t) + along

function opt_scalar_with_store()
    @info "Loading data"
    fname = joinpath(dirname(@__FILE__),"data/Zsac_FLUPS_spectra_params.txt")
    data = readdlm(fname; comments=true)
    t = data[:,1]
    y_exp = data[:,3]
    # keep only between 0 and 2
    m = @. 0 < t < 2
    t = t[m]
    y_exp = y_exp[m]

    guess_e1 = [0.5, 1/0.6, 2.5]
    lb_e1 = [0.4, 1/0.7, 2.45]
    ub_e1 = [0.6, 1/0.2, 2.58]


    # initialize datastore
    popt, pval, sampl, sampl_val = h5open("Zsac_e1.store.h5", "w") do store
        prediction(a0, k0, along) = e1.(t, a0, k0, along)
        e1_s = stored_scalar(prediction, store)
        resid = resid_vs(y_exp, e1_s)
        ssq = ssq_of(resid)
        popt, pval, sampl, sampl_val = surfit_scalar(ssq, guess_e1, lb_e1, ub_e1, 1E-3, 1E-6, 200)
        @info "Lengths:" length(sampl) length(store)
        popt, pval, sampl, sampl_val
    end
    
    return popt, pval, sampl, sampl_val
end

function opt_vect_with_store()
    @info "Loading data"
    fname = joinpath(dirname(@__FILE__), "data/Zsac_FLUPS_spectra_params.txt")
    data = readdlm(fname; comments=true)
    t = data[:,1]
    y_exp = data[:,3]
    # keep only between 0 and 2
    m = @. 0 < t < 2
    t = t[m]
    y_exp = y_exp[m]

    guess_e1 = [0.5, 1/0.6, 2.5]
    lb_e1 = [0.4, 1/0.7, 2.45]
    ub_e1 = [0.6, 1/0.2, 2.58]


    # initialize datastore
    store = h5open("Zsac_e1.store.h5", "w")
    @assert length(store) == 0
    function e1_v(t,  a0::Float64, k0::Float64, along::Float64)
        a0.*exp.(-k0.*t) .+ along
    end
    e1_s = stored(e1_v, store)
    popt, pval, sampl, sampl_val = surfit(e1_s, t, y_exp, guess_e1, lb_e1, ub_e1, 1E-3, 1E-6, 200)
    @info "Lengths:" length(sampl) length(store)
    close(store)

    
    return popt, pval, sampl, sampl_val
end

#pt_vect_with_store()
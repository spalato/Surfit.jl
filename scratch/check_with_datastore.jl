using DelimitedFiles
using Surrogates
using Surfit
using HDF5

e1(t, a0, k0, along) = a0*exp(-k0*t) + along

function opt_with_store()
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
    store = h5open("Zsac_e1.store.h5", "cw")fh
    e1_s = stored(e1, store)
    resid = resid_vs(t, y_exp, e1_s)
    ssq = ssq_of(resid)
    ret = surrogatefit(ssq, guess_e1, lb_e1, ub_e1, 1E-3, 1E-6, 200)
end
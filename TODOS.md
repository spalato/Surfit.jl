Code cleanup
- [ ] Refactor surfit to better separate the different elements: initialization, general convergence, small-step refinement (currently: adding a simplex to the grid)
- [ ] Refactor surfit_scalar to benefit from the above

Functionnality
- [ ] Refine simplex usage: check we're still in.
- [ ] Add more benchmark cases types: polynomials, ?? differential equations?
- [ ] Benchmarking battery: Systematic test variation to gather performance statistics
- [ ] Tune algorithm, if possible.

Improvements
- [ ] Add defaults for x_tol, f_tol, f_calls.
- [ ] Compare gaussian approximation of the confidence interval obtained from the surrogate to those of LM (curvature around the minimum)
- [ ] Add in F-test confidence interval? Compare results of surrogate to LM estimation.




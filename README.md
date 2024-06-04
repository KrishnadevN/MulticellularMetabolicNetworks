# MulticellularMetabolicNetworks

## Sampling

Sampling of the flux space is performed using a C++ code provided in the `Sampling` directory. The input files corresponding to the spatial distribution of cells used for simulations is also provided in the same directory. The C++ source could be compiled using a modern compiler, like the one from GCC, as:

    g++ sample.cpp -o sample.out

The sampling could be performed for different parameter values $\beta_G$ and $\beta_O$ as:

    ./sample.out <βG_VALUE> <βO_VALUE> <NPOINTS> <SAVEFILE>

which will sample `NPOINTS` points and save the sampled points to a plain text file `SAVEFILE`. Each line of the saved file corresponds to one sampled point, and would contain the three fluxes – glucose, oxygen and lactate – for all cells, grouped by flux type, i.e., for `N` cells, each line is

    gluc_1 gluc_2 ... gluc_N ox_1 ox_2 ... ox_N lact_1 lact_2 ... lact_N

## Mean-field approximation

The mean-field approximation is implemented using a python code provided in the `Meanfield` directory. The required packages are `numpy` and `numba`, and a file `requirements.txt` is also provided.

With `meanfield.py` in `pythonpath`, one could import the function `meanfield` and pass the values of $\beta_G$ and $\beta_O$ to calculate the values of the partition function, its derivatives, and all relevant fluxes. The inputs could be scalar or an array of values at which the calculation is to be done. For multiple values, the input parameters could be passed as arrays of matching shapes, and the results will have the same shape as the inputs. If any of the parameters is to be held constant, that parameter could also be passed as a scalar. **NOTE:** It is recommended that a small noise is added to the values to prevent any the denominators ($\beta_G$, $\beta_O$, $\beta_1$ and $\beta_2$) in the expressions from evaluating to `0`.

    from meanfield import meanfield

    noise = 0.001357

    beta_g = 0. + noise
    beta_o = np.arange(-5., 5., 0.1 + noise) + noise

    vals = meanfield(beta_g, beta_o)
    part, gluc, ox, lact = vals.Zp, vals.ug, vals.uo, vals.ul

For evaluating quantities in the $\beta_O$-$\beta_G$ plane, pass the values of parameters as one-dimensional arrays, with an additional argument `grid=True`.

    beta_g = np.arange(-12., 2., 0.1 + noise) + noise
    beta_o = np.arange(- 5., 5., 0.1 + noise) + noise

    vals = meanfield(beta_g, beta_o, grid=True)

**NOTE:** The function defaults to `grid=True` if both the inputs are one-dimensional arrays of non-matching lengths.


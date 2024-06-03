# MulticellularMetabolicNetworks

## Sampling

Sampling of the flux space is performed using a C++ code provided in the `Sampling` directory. The input files corresponding to the spatial distribution of cells used for simulations is also provided in the same directory. The C++ source could be compiled using a modern compiler, like the one from GCC, as:

    g++ sample.cpp -o sample.out

The sampling could be performed for different parameter values $\beta_G$ and $\beta_O$ as:

    ./sample.out <βG_VALUE> <βO_VALUE> <NPOINTS> <SAVEFILE>

which will sample `NPOINTS` points and save the sampled points to a plain text file `SAVEFILE`. Each line of the saved file corresponds to one sampled point, and would contain the three fluxes – glucose, oxygen and lactate – for all cells, grouped by flux type, i.e.,

    g1 g2 ... gN o1 o2 ... oN l1 l2 ... lN

## Mean-field approximation


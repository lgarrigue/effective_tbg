## How to run the code
Open Julia with
```
julia -t 8
```
to obtain 8 threads (one can choose another integer depending on the number of thread one can launch). Call
```
include("apply_graphene.jl")
```
to produce the monolayer functions u1, u2, V and Vint. Call
```
include("apply_effpot_and_bands.jl")
```
to produce the effective potentials and compute the bands of the effective Hamiltonian.  
Computation of Vint and band diagrams production are the two steps requiring more ressources, hence they are paralellized with Threads.@threads on the CPU. COMPUTATION OF Vint IS NOT YET PARALLELIZED BECAUSE OF A PROBLEM IN PARALELLIZATION OF DFTK SCF's

## Organization of the code

### Scripts to compute the monolayer functions
**graphene.jl** computes the potentials and Dirac eigenfunctions of the monolayer graphene, and the KS potential of the shifted bilayer  
**apply_graphene.jl** applies the functions of the previous script, with some definite parameters  

### Scripts to compute effective potentials and bands
**effective_potentials.jl** computes the effective potentials  
**band_diagrams_bm_like.jl** computes the band diagrams  
**apply_effpot_and_bands.jl** applies one of the previous two scripts with definite parameters. Applying *effective_potentials.jl* will create the plots of the potentials in cartesian coordinates while applying *band_diagrams_bm_like.jl* will compute the band diagrams

### Secondary scripts
**misc/lobpcg.jl** is the LOBPCG solver taken from DFTK  
**misc/create_bm_pot.jl** enables to create the true Bistritzer-MacDonald potential, for comparisions purposes  
**common_functions.jl** are low-level functions shared by all the main scripts

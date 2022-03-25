## How to run the code
Open Julia, call 

```
include("apply_graphene.jl")
```
to produce the monolayer functions and Vint, and then call
```
include("apply_effpot_and_bands.jl")
```
to produce the effective potentials and compute the bands of the effective Hamiltonian

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

graphene.jl computes the potentials and Dirac eigenfunctions of the monolayer graphene, and the KS potential of the shifted bilayer
apply_graphene.jl applies the previous

effective_potential.jl computes the effective potentials
band_diagrams_bm_like.jl computes the band diagrams
apply_effpot_and_bands.jl applies the previous

misc/lobpcg.jl is the LOBPCG solver taken from DFTK \\
misc/create_bm_pot.jl enables to create the true Bistritzer-MacDonald potential

Open Julia, call 

```
include("apply_graphene.jl")
```
to produce the monolayer functions and Vint, and then include("apply_effpot_and_bands.jl") to produce the effective potentials and compute the bands of the effective Hamiltonian

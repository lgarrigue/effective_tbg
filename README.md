## How to run the code
Open Julia with
```
julia -t 16
```
to obtain 16 threads (choose the number of threads you want to run). Call
```
include("apply_graphene.jl")
```
to produce the monolayer functions u1, u2, V and Vint. Call
```
include("apply_effpot.jl")
```
to produce the effective potentials and compute the bands of the effective Hamiltonian. Computation of Vint and band diagrams production are the two steps requiring more ressources, they are paralellized, naturally on DFTK for the first case, and with Threads.@threads on the CPU for band diagrams.

To produce the band diagrams of the effective model and of the BM model, run
```
include("apply_bands.jl")
```

### Installation
In the pkg mode that one can enter with writing ], one can add the necessary packages by
```
add FFTW, DFTK, JLD, LaTeXStrings, Plots, CairoMakie, DelimitedFiles, DataFrames, Optim, BlockArrays
```
The code works with DFTK v0.5.10, with more recent versions it could be needed to update some DFTK functions calls

## Organization of the code

### Scripts to compute the monolayer functions
**graphene.jl** computes the potentials and Dirac eigenfunctions of the monolayer graphene, and the KS potential of the shifted bilayer Vint
**apply_graphene.jl** applies the functions of the previous script, with some definite parameters, and exports the functions. It computes SCF of the monolayer to get the Dirac eigenvalues and the effective potential, verifies their symmetries (2pi/3 rotation, mirror, z -> -z), exports them, and computes Vint which is the long step

### Scripts to compute effective potentials
**effective_potentials.jl** computes the effective potentials
**apply_effpot.jl** applies the previous script with definite parameters. It builds the effective potentials, verifies symmetries on them (particle-hole, translation, parity-time, etc), and plots them
**effpots_against_d.jl** creates the figures for a study of effective potentials quantities against d, corresponding to the article "A simple derivation of moiré-scale continuous models for twisted bilayer graphene"

### Scripts to compute bands
**band_diagrams_bm_like.jl** assembles the effective Hamiltonian and computes the band diagrams  
**apply_bands.jl** applies the previous script functions. It will either create and plot the band diagram for some definite value of angle theta or do a study of the bandwidth, band gap and Fermi velocity with respect to theta

### Secondary scripts
**misc/lobpcg.jl** is the LOBPCG solver taken from DFTK  
**misc/create_bm_pot.jl** enables to create the true Bistritzer-MacDonald potential, for comparisions purposes  
**common_functions.jl** are low-level functions shared by all the main scripts

### Details
**misc/numerical_details.pdf** contains more information about the implementation

### Contact
If you have a question, for instance to run the code, it will be a pleasure to answer you, so please contact me at louis.garrigue@gmail.com

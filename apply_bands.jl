include("band_diagrams_bm_like.jl")
using DelimitedFiles, CairoMakie, LaTeXStrings

#################### Third step : compute the bands diagram

function explore_band_structure_Heff()
    p = Basis()
    ############## Parameters
    p.dim = 2
    # Dimensions
    N = 27; Nz = 600; gauge_param = 1

    # Distances
    interlayer_distance = 6.45
    p.a = 4.66 # fundamental cell length

    # Imports u1, u2, V, Vint, v_fermi and computes the effective potentials
    compute_Vint = true
    EffV = import_and_computes(N,Nz,gauge_param,compute_Vint,interlayer_distance)
    # Reduces the size of the Fourier basis, because the initial dimensions are too high to compute
    reduce_N(EffV,11); p.N = EffV.N

    # Misc
    p.l = 15 # number of eigenvalues we compute for the band diagrams
    job = "diagram" # ∈ ["bandwidths","diagram"], select what to do, either produce band diagrams, or produce the graph with bandwidths/fermi velocities/band gap against θ
    init_basis(p) # does some initializations
    p.resolution_bands = (job=="bandwidths") ? 8 : 50 # number of K-points computed on each segement of the band diagram
    p.folder_plots_bands = "eff" # folder where band diagrams will be produced
    p.solver = "Exact" # ∈ ["Exact","LOBPCG"]
    p.plots_article = true
    p.energy_amplitude = 250 # band diagrams will have a y-axis of amplitude 2*energy_amplitude

    ############## Defines the momentum points of the bands diagram, in reduced coordinates
    p.K1 = [-2,1]/3; p.K2 = [-1,2]/3
    Γ = [0.0,0.0]; Γ2 = 2*p.K1-p.K2; M = Γ2/2
    p.Klist = [p.K2,p.K1,Γ2,M,Γ]; p.Klist_names = ["K_2","K_1","Γ'","M","Γ"]
    # plot_path(p.Klist,p.Klist_names,p) # plots the path used in the band diagram
    if job=="bandwidths" p.Klist = [p.K1,M,Γ]; p.Klist_names = ["K_1","M","Γ"] end

    ############## Defines the effective potentials
    multiply_potentials(p.sqi,EffV)
    # Build SΣ
    SΣ = build_offdiag_V(EffV.Σ,p)
    S = Hermitian(I + SΣ)
    p.ISΣ = Hermitian(inv(sqrt(S)))

    # Shifts the Fermi energy for plots
    # W = EffV.compute_Vint ? EffV.Wplus_tot : EffV.W_V_plus
    # mW = real(mean_block(W,EffV)[1,1]); p.energy_center = mW*hartree_to_ev*1e3 # when the zero of band diagrams is not

    px("Symmetry tests")
    # test_div_JA(EffV) # symmetry test
    offdiag_A_k(EffV.J𝔸1,EffV.J𝔸2,p.K1/4,p;name="J",test=true)
    offdiag_div_Σ_∇(EffV.Σ,p.K1,p;test=true)
    Dirac_k(p.K1,p;test=true)
    ondiag_mΔ_k(p.K1,p;test=true)

    ############## Part of the operator which does not depend on k
    cst_op_new = p.ISΣ*( build_offdiag_V(EffV.𝕍,p;test=true) + build_ondiag_W(EffV.Wplus_tot,EffV.Wminus_tot,p;test=true))*p.ISΣ
    
    ############## Part of the operator which depends on k
    Kf_pure(K) = p.vF*Dirac_k(K,p)
    Kf_new(K,c,ε) = p.ISΣ*(
                            c*p.vF*Dirac_k(K,p)
                            + c*offdiag_A_k(EffV.J𝔸1,EffV.J𝔸2,K,p)
                            + (1/2)*ε*(p.vF*Dirac_k(K,p;coef_1=-1,J=true)
                                       + ondiag_mΔ_k(K,p)
                                       + offdiag_div_Σ_∇(EffV.Σ,K,p)
                                       )
                            )*p.ISΣ

    ############## Choice for values of θ, in case job == "bandwidths"
    # θs stores values of θ for which we want to plot the graphs, to increase its resolution at critical points
    θs_min_fv_bm_110 = [1.175,0.545,0.535,0.518]
    θs_min_bw_new = [1.160,0.446]
    θs_min_fv_bm_126 = [0.594,0.490,0.477,0.4060]
    θs_min_fv_new = [0.5295,0.525,0.5188]
    special_θs = vcat(θs_min_bw_new,θs_min_fv_new,θs_min_fv_bm_126,θs_min_fv_bm_110)

    # Function which creates the band diagram
    bands_partial(kind,θ) = bands(kind,θ,p,EffV;parity_band=false,cst_op=cst_op_new,Kf_new=Kf_new,Kf_pure=Kf_pure) 

    ############## Launches the functions
    if job=="bandwidths"
        θs = sort(vcat((0.4:0.0002:1.4),special_θs))
        compute_bandwidths_and_velocity(θs,p,bands_partial;alleviate="all")
    elseif job=="diagram"
        create_band_diagrams(θs_min_fv_bm_110[1],θs_min_bw_new[1],"all",p,bands_partial)
    end
    p
end

p = explore_band_structure_Heff()
nothing

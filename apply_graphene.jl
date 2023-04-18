include("graphene.jl")

#################### First step : produce Bloch eigenfunctions of the 3d monolayer, Vks and Vint. The long step is the computation of Vint

function produce_bloch_functions_and_potentials()
    #################### Parameters
    p = Params()
    # Fixed parameters
    p.dim = 3 # stores all we need to store
    p.a = 4.66 # length of the vectors of the fundamental cell, in Bohr
    p.i_state = 4 # u1 will be the i^th eigenmode (i_state=1 is the ground state), u2 the (i+1)^th, u0 the (i-1)^th

    # Changeable monolayers parameters
    p.gauge_param = 1 # gauge_param ∈ ±1, so that <Φ1,(-i∇_x)Φ2> = gauge_param*vF*[1,-i]. Take 1 to be coherent with the bands diagram implementation
    # L : periodicity in z (both for mono and bilayer computations)
    # ecut_over_kd2 : ecut/(kd^2)
    ecut_over_kd2 = 20; p.L = 50
    ecut_over_kd2 = 41; p.L = 115

    p.kgrid = [5,5,1] # for computing the Kohn-Sham potential
    p.tol_scf = 1e-4

    # Parameters on Vint
    compute_Vint = true
    p.Nint = 3 # resolution (in axis X and Y) of the disregistries grid for computing Vint. Everything is very not dependent of Nint, one can take just 3 (even 2)
    # d_list = [6.45] # list of values of d (interlayer distance) for which we compute Vint
    d_list = vcat([6.45],[0.01],(0.1:0.1:11)) # computes different Vint to prepare the study of effective potentials measures against d
    d_list = [6.45]

    # Others
    p.plots_cutoff = 3 # cutoff for plots, values of k will be [-plots_cutoff,plots_cutoff]^d
    p.export_plots_article = false # whether we produce plots for the article "A simple derivation of moiré-scale continuous models for twisted bilayer graphene"
    p.alleviate = false # chooses parameters that eases the computation ressources and computaton time
    if p.alleviate
        p.L = 30
        p.kgrid = [4,4,1]
        ecut_over_kd2 = 20
        p.tol_scf = 1e-3
        p.Nint = 1
        d_list = [6.45]
    end
    p.ecut = ecut_over_kd2*norm_K_cart(p.a)^2
    px("(ecut/kD^2,L)=(",ecut_over_kd2,",",p.L,") ; kgrid=",p.kgrid,") ; gauge_param=",p.gauge_param)

    # Initialization of some parameters
    init_params(p)

    #################### Computations
    # SCF for the monolayer
    scfres = scf_graphene_monolayer(p)

    # Computes the Dirac Bloch functions u1, u2 (at the Dirac point)
    get_dirac_eigenmodes(p)
    get_natural_basis_u1_and_u2(p)
    test_normalizations(p)
    extract_nonlocal(p)

    # Computes the non local contribution of the Fermi velocity
    non_local_deriv_energy(4,p)

    # Computes the Fermi velocity
    # get_fermi_velocity_with_finite_diffs(4,p) # Computing dE/dk with diagonalizations of H(k), should get 0.380 or 0.381
    records_fermi_velocity_and_fixes_gauge(p)
    # fermi_velocity_from_scalar_products(p) # displays <ϕ_p,(-i∇) ϕ_j> and <ϕ_p,(-i∇) P ϕ_j>, where P is the parity operator
    # double_derivatives_scalar_products(p) # <∂i ϕk,∂j ϕp> (i,j ∈ {1,2}) and <∂z ϕk,∂z ϕp>

    # Symmetry tests
    test_rot_sym(p)
    test_mirror_sym(p)

    # Exports v, u1, u2, φ and the non local coefficient
    exports_v_u1_u2_φ(p)

    # Plots V, u0, u1, u2, non_local_φ
    if true
        px("Makes plots")
        resolution = 50
        n_motifs = 3
        # rapid_plot(p.u1_fc,p;n_motifs=n_motifs,name="ϕ1",res=resolution,bloch_trsf=true)
        # rapid_plot(p.u2_fc,p;n_motifs=n_motifs,name="ϕ2",res=resolution,bloch_trsf=true)
        # rapid_plot(p.v_monolayer_fc,p;n_motifs=n_motifs,name="v",res=resolution,bloch_trsf=false)
        # rapid_plot(p.non_local_φ1_fc,p;n_motifs=n_motifs,name="non_local_φ1",res=resolution,bloch_trsf=true)
    end

    # Prints wAA
    p.interlayer_distance = 6.45
    (wAA,wC) = get_wAA_wC_from_monolayer(p.v_monolayer_dir,p)

    # Computes Vint (expensive in time)
    if compute_Vint
        for d in d_list
            p.interlayer_distance = d # distance between the two layers
            px("Computes Vint for d=",d,", Nint=",p.Nint)
            px("Computes the Kohn-Sham potential of the bilayer at each disregistry (long step): ",p.Nint,"×",p.Nint,"=",p.Nint^2," steps")
            # Computes all the Kohn-Sham potentials with SCF (long step)
            compute_V_bilayer_Xs(p)
            # Computes Vint(Xs,z)
            Vint_Xs_fc = compute_Vint_Xs(p)
            # Computes Vint(z)
            p.Vint_f = form_Vint_from_Vint_Xs(Vint_Xs_fc,p)
            # Computes the dependency of Vint_Xs on Xs (the in-plane variable)
            computes_δ_Vint(Vint_Xs_fc,p.Vint_f,p)
            # Plots, exports, tests
            p.Vint_dir = real.(myifft(p.Vint_f,p.L))
            test_z_parity(p.Vint_dir,1,p;name="Vint")
            export_Vint(p)
            plot_Vint(p)
        end
    end
    p
end

p = produce_bloch_functions_and_potentials()
nothing

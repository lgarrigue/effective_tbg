include("graphene.jl")

#################### First step : produce Bloch eigenfunctions of the 3d monolayer, Vks and Vint. The long step is the computation of Vint

function produce_bloch_functions_and_potentials()
	p = Params() # stores all we need to store
	# Choose parameters
	p.a = 4.66 # length of the vectors of the fundamental cell
	p.interlayer_distance = 6.45 # distance between the two layers
	p.L = 20 # periodicity in z (both for mono and bilayer computations)
	p.i_state = 4 # u1 will be the i^th eigenmode, u1 the (i+1)^th, u0 the (i-1)^th
	p.ecut = 2 # DFTK's ecut
	p.kgrid = [10,10,1] # for computing the KS potential
	p.tol_scf = 1e-3
	p.plots_cutoff = 7 # Fourier cutoff for plots
	init_params(p)
	compute_Vint = true

	px("Computes the Kohn-Sham potential of the monolayer")
	scfres = scf_graphene_monolayer(p)

	# Computes the Dirac Bloch functions u1, u2 (at the Dirac point)
	get_dirac_eigenmodes(p)

	# Rotates u1 and u2 in U(2) to obtain the symmetric ones
	rotate_u1_and_u2(p)

	# Computes the Fermi velocity
	# get_fermi_velocity_with_finite_diffs(p) # Computing dE/dk with diagonalization
	# fermi_velocity_from_rotated_us(p) # Doing scalar products

	# Symmetry tests
	if true
		test_rot_sym(p)
		test_z_parity(p.u1_dir,-1,p;name="u1")
		test_z_parity(p.u2_dir,-1,p;name="u2")
		test_x_parity(abs.(p.u1_dir),p;name="|u1|")
		test_x_parity(abs.(p.u0_dir),p;name="|u0|")
		test_z_parity(p.v_monolayer_dir,1,p;name="v")
		test_x_parity(p.v_monolayer_dir,p;name="v")
	end

	# Exports v, u1, u2
	exports_v_u1_u2(p)

	# Plots u0, u1, u2
	resolution = 75
	n_motifs = 15
	rapid_plot(p.u0,p;n_motifs=n_motifs,name="u0",res=resolution)
	rapid_plot(p.u1,p;n_motifs=n_motifs,name= "u1",res=resolution)
	# rapid_plot(R(p.u1,p),p;n_motifs=n_motifs,res=resolution,name="u2",bloch_trsf=false)

	# Computes Vint (expensive in time)
	if compute_Vint
		px("Computes the Kohn-Sham potential of the bilayer at each disregistry (long step): ",p.N,"×",p.N,"=",p.N2d," steps")
		V_bil_Xs = hat_V_bilayer_Xs(p)
		# Computes Vint(Xs,z)
		Vint_Xs = compute_Vint_Xs(V_bil_Xs,p)
		# Computes Vint(z)
		Vint_f = form_Vint_from_Vint_Xs(V_bil_Xs,p)
		# Plots, exports, tests
		plot_VintXs_Vint(Vint_Xs,Vint_f,p)
		export_Vint(Vint_f,p)
		test_z_parity(ifft(Vint_f),1,p;name="Vint")
	end
end

produce_bloch_functions_and_potentials()

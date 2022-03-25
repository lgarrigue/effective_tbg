include("graphene.jl")

#################### First step : produce Bloch eigenfunctions of the 3d monolayer, Vks and Vint. The long step is the computation of Vint

function produce_bloch_functions_and_potentials()
	p = init_params()

	# Choose parameters
	p.i_state = 4
	p.ecut = 3
	p.kgrid = [10,10,1]
	p.tol_scf = 1e-5
	p.cutoff_plots = 7
	compute_Vint = false

	# Solves
	scfres = scf_graphene_monolayer(p)
	get_dirac_eigenmodes(p)
	rotate_u1_and_u2(p)
	# get_fermi_velocity_with_finite_diffs(p)
	# fermi_velocity_from_rotated_us(p)

	# Tests
	if true
		test_rot_sym(p)
		test_z_parity(p.u1_dir,-1,p;name="u1")
		test_z_parity(p.u2_dir,-1,p;name="u2")
		test_z_parity(p.v_monolayer,1,p;name="Vks")
	end

	# Exports
	exports_v_u1_u2(p)

	# Plots
	resolution = 75
	n_motifs = 15
	rapid_plot(p.u0,p;n_motifs=n_motifs,res=resolution,name="u0")
	rapid_plot(p.u1,p;n_motifs=n_motifs,res=resolution,name= "u1")
	# rapid_plot(R(p.u1,p),p;n_motifs=n_motifs,res=resolution,name="u2",bloch_trsf=false)

	# Computes Vint (expensive in time)
	if compute_Vint
		V_bil_Xs = hat_V_bilayer_Xs(p)
		Vint_Xs = compute_Vint_Xs(V_bil_Xs,p)
		Vint_f = compute_Vint(V_bil_Xs,p)
		plot_VintXs_Vint(Vint_Xs,Vint_f,p)
		export_Vint(Vint_f,p)
		test_z_parity(ifft(p.Vint_f),1,p;name="Vint")
	end
end

produce_bloch_functions_and_potentials()

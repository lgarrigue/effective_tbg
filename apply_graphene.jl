include("graphene.jl")

#################### First step : produce Bloch eigenfunctions of the 3d monolayer, Vks and Vint. The long step is the computation of Vint

function produce_bloch_functions_and_potentials()
	p = Params() # stores all we need to store
	# Choose parameters
	p.ecut = 5 # DFTK's ecut, convergence of u's for ecut ≃ 15
	px("ecut ",p.ecut)
	p.a = 4.66 # length of the vectors of the fundamental cell
	p.interlayer_distance = 6.45 # distance between the two layers
	p.L = 20 # periodicity in z (both for mono and bilayer computations)
	p.i_state = 4 # u1 will be the i^th eigenmode, u1 the (i+1)^th, u0 the (i-1)^th
	p.kgrid = [10,10,1] # for computing the KS potential
	p.tol_scf = 1e-4
	p.plots_cutoff = 3 # Fourier cutoff for plots
	init_params(p)
	compute_Vint = false

	px("Computes the Kohn-Sham potential of the monolayer")
	scfres = scf_graphene_monolayer(p)

	# Computes the Dirac Bloch functions u1, u2 (at the Dirac point)
	get_dirac_eigenmodes(p)

	# Rotates u1 and u2 in U(2) to obtain the symmetric ones
	rotate_u1_and_u2(p)

	# Extracts the non local terms
	extract_nonlocal(p)
	# Computes the non local contribution of the Fermi velocity
	non_local_deriv_energy(4,p)

	(Mu1,Tu1) = (M(p.u0_fb,p),τ(p.u0_fb,p.shift_K,p))
	px("MΦ0 = Φ0 ",norm(Mu1.-Tu1)/norm(Mu1))

	# Tests normalization
	px("Normalization of u1 Fourier: ",norms(p.u1_fc,p))
	px("Normalization of u1 direct: ",norms(p.u1_dir,p,false))
	px("Orthonormality |<u1,u2>|= ",abs(scaprod(p.u1_fc,p.u2_fc,p)) + abs(scaprod(p.u1_dir,p.u2_dir,p,false)))
	px("Potential energy of u1 <u1,V u1> in Fourier: ",scaprod(p.u1_fc,cyclic_conv(p.u1_fc,p.v_monolayer_fc),p))

	# Computes the Fermi velocity
	# p.v_fermi = get_fermi_velocity_with_finite_diffs(4,p) # Computing dE/dk with diagonalizations of H(k), should get 0.380 or 0.381
	# fermi_velocity_from_rotated_us(p) # Doing scalar products

	# Fermi velocity from derivation
	(∂1_u2_f,∂2_u2_f,∂3_u2_f) = ∇(p.u2_fc,p)
	c1 = im*scaprod(p.u1_fc,∂1_u2_f,p,true); c2 = im*scaprod(p.u1_fc,∂2_u2_f,p,true)
	px("i<u1,∇r u2> = [",c1,",",c2,"] ; ratio ",c1/c2," ; |c1|=",abs(c1)," ; |c2|=",abs(c2))
	p.v_fermi = abs(c1)

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

	# Exports v, u1, u2, φ and the non local coefficient
	exports_v_u1_u2_φ(p)

	# Plots u0, u1, u2, φ
	px("Makes plots")
	resolution = 30
	n_motifs = 2
	# rapid_plot(p.u0_fc,p;n_motifs=n_motifs,name="ϕ0",res=resolution,bloch_trsf=true)
	# rapid_plot(p.u1_fc,p;n_motifs=n_motifs,name="ϕ1",res=resolution,bloch_trsf=true)
	# if the plot of u1 is small, this is probably because of it's antisymmetry in z, and because the dominating Fourier transform coefficients are away from 0 and cut by plot_cutoff
	# rapid_plot(p.u2_fc,p;n_motifs=n_motifs,name="ϕ2",res=resolution,bloch_trsf=true)
	# rapid_plot(p.v_monolayer_fc,p;n_motifs=n_motifs,name="v",res=resolution,bloch_trsf=false)
	# rapid_plot(p.non_local_φ_fc,p;n_motifs=n_motifs,name="non_local_φ",res=resolution,bloch_trsf=true)

	# Computes Vint (expensive in time)
	if compute_Vint
		px("Computes the Kohn-Sham potential of the bilayer at each disregistry (long step): ",p.N,"×",p.N,"=",p.N2d," steps")
		# p.V_bilayer_Xs_fc = randn(p.N,p.N,p.Nz)
		compute_V_bilayer_Xs(p)
		# Computes Vint(Xs,z)
		Vint_Xs_fc = compute_Vint_Xs(p)
		# Computes Vint(z)
		p.Vint_f = form_Vint_from_Vint_Xs(Vint_Xs_fc,p)
		# Computes the dependency of Vint_Xs on Xs
		computes_δ_Vint(Vint_Xs_fc,p.Vint_f,p)
		# Plots, exports, tests
		p.Vint_dir = real.(myifft(p.Vint_f))
		test_z_parity(p.Vint_dir,1,p;name="Vint")
		export_Vint(p)
		plot_Vint(p)
	end
	p
end

function test_convergence_u1()
	ecuts = [30,40,50]
	N0,Nz0,N0d,Nz0d = 0,0,0,0
	u1s = []; u2s = []
	for j=1:length(ecuts)
		p = Params() # stores all we need to store
		# Choose parameters
		p.ecut = ecuts[j]
		p.a = 4.66
		p.interlayer_distance = 6.45
		p.L = 20 
		p.i_state = 4
		p.kgrid = [10,10,1]
		p.tol_scf = 1e-4
		p.plots_cutoff = 3
		init_params(p)
		scfres = scf_graphene_monolayer(p)
		get_dirac_eigenmodes(p)
		rotate_u1_and_u2(p)
		test_rot_sym(p)
		ref1 = p.u1_fc
		ref2 = p.u2_fc
		px("norms ",norm(p.u0_fb)," ",norm(p.u1_fb))
		if j==1
			N0 = p.N
			Nz0 = p.Nz
			N0d = floor(Int,N0/2)
			Nz0d = floor(Int,Nz0/2)
		end
		u1 = zeros(ComplexF64,N0,N0,Nz0)
		u2 = zeros(ComplexF64,N0,N0,Nz0)
		for mi=-N0d:N0d, mj=-N0d:N0d, ml=-Nz0d:Nz0d
			u1[k_inv_1d(mi,N0),k_inv_1d(mj,N0),k_inv_1d(ml,Nz0)] = ref1[k_inv_1d(mi,p.N),k_inv_1d(mj,p.N),k_inv_1d(ml,p.Nz)]
			u2[k_inv_1d(mi,N0),k_inv_1d(mj,N0),k_inv_1d(ml,Nz0)] = ref2[k_inv_1d(mi,p.N),k_inv_1d(mj,p.N),k_inv_1d(ml,p.Nz)]
		end
		s = (2000,1000)
		mul = 1e3
		fun = real
		Z = 5
		h1 = heatmap(mul*fun.(u1[:,:,Z]),size=s)
		h2 = heatmap(mul*fun.(ref1[:,:,Z]),size=s)
		pl = plot(h1,h2)
		savefig(pl,string(ecuts[j],".png"))

		testZ1 = [sum(abs.(ref1[:,:,z])) for z=1:p.Nz] # check that Z is not such that u is small
		pZ = savefig(plot(testZ1),"testZ1.png")

		testZ0 = [sum(abs.(p.u0_fc[:,:,z])) for z=1:p.Nz] # check that Z is not such that u is small
		pZ = savefig(plot(testZ0),"testZ0.png")
		M = p.N^2*p.Nz
		push!(u1s,u1)
		push!(u2s,u2)
	end
	px("Distances")
	# a = [abs.(u1s[i]) for i=1:length(u1s)]
	a = [u1s[i].*u2s[i] for i=1:length(u1s)] # converges
	a = [conj.(u1s[i]).*u2s[i] for i=1:length(u1s)] # does not converge
	for i=1:length(u1s)-1
		px(sum(abs.(a[i+1].-a[i])/sum(abs.(a[i+1]))))
	end
end

p = produce_bloch_functions_and_potentials()
# test_convergence_u1()
nothing

# Gauge invariance
# U = \mat{e^{i\theta} & 0 \\ 0 & e^{-i\theta}}
# U^* H U = \mat{W^+ & e^{-i2\theta} V \\ e^{i2\theta} V^* & W^-}
# and doing ϕ1 -> ϕ1 e^{i\theta}, we recover H

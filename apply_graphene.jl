include("graphene.jl")

#################### First step : produce Bloch eigenfunctions of the 3d monolayer, Vks and Vint. The long step is the computation of Vint


# (L=30) waa ecut/kd^2 35 -> 104 ; 40 -> 103 ; 45 -> 103
# (ecut/kd^2=40) L=30 -> 103 ; 40 -> 113 ; 50 -> 119 ; 60 -> 123 ; 70 -> 126 ; 80 -> 128 ; 90 -> 130 ; 100 -> 131 ; 110 -> 132 ; 120 -> 133 ; 130 -> 133.6 killed for saving ; 140 -> 134.3 ; 150 : killed before computing wAA
# L=100, ecut/kd^2 = 40 ; tol_scf = 1e-6 -> wAA = 130.9 ; 1e-5 -> 130.9 ; 1e-4 -> 130.9 ; 1e-3 -> 130.6 ; 1e-2 -> 130.6 but other quantities like u1 are not as precise
# L=60, ecut/kd^2 = 40, tol_scf = ;; kgrid = [8,8,1] -> wAA = 122.9 ; [7,7,1] -> 122.8 ; [6,6,1] -> 123.1 ; [5,5,1] -> 123.1 ; [4,4,1] -> 123.2 ; [3,3,1] -> 123.6


# kgrid = [5,5,1]; tol = 1e-4
# (ecut/kd^2,L) (N,Nz) -> wAA_v
#
# (25,100) (24,432) -> 145.0
# (25,120) (24,500) -> 145.3
# (25,140) (24,576) -> 145.5
# (25,160) (24,675) -> 145.6
# (25,180) (24,750) -> 145.8
#
# (25,120) (24,500) -> 145.3
# (30,120) (24,540) -> 143.7
# (35,120) (25,600) -> 143.0
# (40,120) (25,625) -> 142.0
# (41,120) (27,625) -> 141.9
# (45,120) (27,675) -> killed
#
# (25,100) (24,432) -> 145.0
# (35,100) (25,486) -> 142.9
# (40,100) (25,540) -> 141.77
# (45,100) (27,576) -> 141.75
# (50,100) (30,576) -> killed
#
# (40, 40) (25,216) -> 140.8
# (40, 60) (25,320) -> 141.3
# (40, 80) (25,432) -> 141.7
# (40,100) (25,540) -> 141.77
# (40,120) (25,625) -> 142.0
#
# (50, 60) (30,360) -> 141.14
# (50, 80) (30,480) -> 141.34
# (50,100) (30,576) -> killed
#
# (35,160) (25,800) -> killed
# (41,115) (27,600) -> 141.8 <---

# Quantities converged for ecut/kd^2 ≃ 40, L ≃ 125 (not really); tol_scf ≃ 1e-3; kgrid=[5,5,1]
function produce_bloch_functions_and_potentials()
	p = Params()

	# Fixed parameters
	p.dim = 3 # stores all we need to store
	p.a = 4.66 # length of the vectors of the fundamental cell
	p.i_state = 4 # u1 will be the i^th eigenmode, u1 the (i+1)^th, u0 the (i-1)^th

	# Changeable monolayers parameters
	p.L = 115 # periodicity in z (both for mono and bilayer computations)
	ecut_sur_kd2 = 41
	p.ecut = ecut_sur_kd2*norm_K_cart(p.a)^2; px("ecut ",p.ecut) # DFTK's ecut, convergence of u's for ecut ≃ 15
	p.kgrid = [5,5,1] # for computing the KS potential
	p.tol_scf = 1e-4
	px("(ecut/kD^2,L)=(",ecut_sur_kd2,",",p.L,")")

	# Params Vint
	compute_Vint = true
	p.Nint = 3 # everything is very not dependent of Nint, one can take just 3 (even 2)
	d_list = vcat([0.01],(0.1:0.1:11))#,[6.45])
	# d_list = (7.7:0.1:11)#,[6.45])
	d_list = [6.45]

	# Misc
	p.plots_cutoff = 3 # Fourier cutoff for plots
	p.export_plots_article = true
	p.alleviate = false

	# Init
	init_params(p)

	if p.alleviate
		p.L = 30
		p.kgrid = [4,4,1]
		p.ecut = 30
		p.tol_scf = 1e-3
		p.Nint = 1
	end

	scfres = scf_graphene_monolayer(p)
	# Computes the Dirac Bloch functions u1, u2 (at the Dirac point)
	get_dirac_eigenmodes(p)
	get_natural_basis_u1_and_u2(p)
	extract_nonlocal(p)
	test_rot_sym(p)
	test_mirror_sym(p)
	# Computes the non local contribution of the Fermi velocity
	# non_local_deriv_energy(4,p)
	# plot_mean_V(p)
	
	p.interlayer_distance = 6.45
	(wAA,wC) = get_wAA_wC_from_monolayer(p.v_monolayer_dir,p)

	px("norm u Four ",norm(p.u1_fc))

	test_scaprod_fft_commutation(p)
	# Tests normalization
	px("Normalization of u1 Fourier: ",norms(p.u1_fc,p))
	px("Normalization of u1 direct: ",norms(p.u1_dir,p,false))
	px("Orthonormality |<u1,u2>|= ",abs(scaprod(p.u1_fc,p.u2_fc,p)) + abs(scaprod(p.u1_dir,p.u2_dir,p,false)))
	px("Potential energy of u1 <u1,V u1> in Fourier: ",scaprod(p.u1_fc,cyclic_conv(p.u1_fc,p.v_monolayer_fc,p.Vol),p))

	# Computes the Fermi velocity
	# p.v_fermi = get_fermi_velocity_with_finite_diffs(4,p) # Computing dE/dk with diagonalizations of H(k), should get 0.380 or 0.381
	# fermi_velocity_from_rotated_us(p) # Doing scalar products
	records_fermi_velocity_and_fixes_gauge(p)

	# p.u2_fb .*= -1
	# p.u2_dir = G_to_r(p.basis,p.K_kpt,p.u2_fb)
	# p.u2_fc = myfft(p.u2_dir,p.Vol)

	# Exports v, u1, u2, φ and the non local coefficient
	exports_v_u1_u2_φ(p)

	# Plots u0, u1, u2, φ
	px("Makes plots")
	resolution = 10
	n_motifs = 2
	# rapid_plot(p.u0_fc,p;n_motifs=n_motifs,name="ϕ0",res=resolution,bloch_trsf=true)
	# rapid_plot(p.u1_fc,p;n_motifs=n_motifs,name="ϕ1",res=resolution,bloch_trsf=true)
	# if the plot of u1 is small, this is probably because of it's antisymmetry in z, and because the dominating Fourier transform coefficients are away from 0 and cut by plot_cutoff
	# rapid_plot(p.u2_fc,p;n_motifs=n_motifs,name="ϕ2",res=resolution,bloch_trsf=true)
	# rapid_plot(p.v_monolayer_fc,p;n_motifs=n_motifs,name="v",res=resolution,bloch_trsf=false)
	# rapid_plot(p.non_local_φ_fc,p;n_motifs=n_motifs,name="non_local_φ",res=resolution,bloch_trsf=true)

	# Computes Vint (expensive in time)
	if p.alleviate d_list = [6.45] end
	if compute_Vint
		for d in d_list
			p.interlayer_distance = d # distance between the two layers
			px("Computes Vint for d=",d, "Nint=",p.Nint)
			px("Computes the Kohn-Sham potential of the bilayer at each disregistry (long step): ",p.Nint,"×",p.Nint,"=",p.Nint^2," steps")
			# p.V_bilayer_Xs_fc = randn(p.N,p.N,p.Nz)
			compute_V_bilayer_Xs(p)
			# Computes Vint(Xs,z)
			Vint_Xs_fc = compute_Vint_Xs(p)
			# Computes Vint(z)
			p.Vint_f = form_Vint_from_Vint_Xs(Vint_Xs_fc,p)
			# Computes the dependency of Vint_Xs on Xs
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
#
# d : reprendre à 5.2

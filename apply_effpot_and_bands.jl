include("band_diagrams_bm_like.jl")

#################### Second step : compute the effective potentials 𝕍, 𝕎, 𝔸, etc. Very rapid

function computes_and_plots_effective_potentials()
	# Parameters
	N = 8; Nz = 27
	# N = 9; Nz = 36
	# N = 12; Nz = 45
	# N = 15; Nz = 50
	p = EffPotentials()
	p.plots_cutoff = 7
	p.plots_res = 50
	p.plots_n_motifs = 10
	take_Vint_into_account = false
	produce_plots = true

	# Initializations
	import_u1_u2_V(N,Nz,p)
	import_Vint(p)
	init_EffPot(p)
	px("Test norm ",norms3d(p.u1_dir,p,false)," and in Fourier ",norms3d(p.u1_f,p))

	# True BM potential
	α = 0.5; β = 1.0
	T = build_BM(α,β,p)

	if false # tests Cm_s
		P1 = build_potential_direct(p.u1v_f,p.u1_f,p)
		P2 = ifft(build_potential(p.u1v_f,p.u1_f,p))
		Cm = build_Cm(p.u1_f,p.u1_f,p)
		rapid_plot([P1,P2],real,p)
		display(Cm)
	end

	build_blocks_potentials(p) # computes Wplus, 𝕍_V and Σ
	build_block_𝔸(p) # computes 𝔸
	to_test = p.Wplus[1,1]


	px("\nW_Vint matrix\n",p.W_Vint_matrix,"\n")

	px("|<u1,u2>| = ",abs(sca3d(p.u1_dir,p.u2_dir,p,false)))
	(∂1_u2_f,∂2_u2_f,∂3_u2_f) = ∇(p.u2_f,p)
	c1 = 2im*sca3d(p.u1_f,∂1_u2_f,p,true)
	c2 = 2im*sca3d(p.u1_f,∂2_u2_f,p,true)
	# c3 = 2im*sca3d(p.u1_f,∂3_u2_f,p,true)
	px("2i<u1,∇r u2> = [",c1,",",c2,"] ; ratio ",c1/c2," ; |c1|=",abs(c1)," ; |c2|=",abs(c2)," ; should be 0.42")

	test_z_parity(p.u1_dir,-1,p;name="u1")
	test_z_parity(p.u2_dir,-1,p;name="u2")
	test_z_parity(p.v_dir,1,p;name="Vks")

	W = p.Wplus
	V = p.𝕍_V
	if take_Vint_into_account
		import_Vint(p)
		px("Matrix Vint for W: ",p.W_Vint_matrix)
		plot_block_cart(p.𝕍_Vint,p;title="Vint")
		W = p.W
		V = p.𝕍
	end

	# Particle-hole
	px("\nTests particle-hole symmetry")
	test_particle_hole_block(T,p;name="T")
	test_particle_hole_block(V,p;name="V")
	test_particle_hole_block(W,p;name="W")
	test_particle_hole_block(p.Σ,p;name="Σ")
	test_particle_hole_block(p.𝔸1,p;name="A1")
	test_particle_hole_block(p.𝔸2,p;name="A2")

	# Partiy-time
	px("\nTests PT symmetry")
	test_PT_block(T,p;name="T")
	test_PT_block(W,p;name="W")
	test_PT_block(V,p;name="V")
	test_PT_block(p.𝔸1,p;name="A1")
	test_PT_block(p.𝔸2,p;name="A2")
	test_PT_block(p.Σ,p;name="Σ")

	# Mirror
	px("\nTests mirror symmetry")
	test_mirror_block(T,p;name="T",herm=true)
	test_mirror_block(W,p;name="W",herm=true)
	test_mirror_block(W,p;name="W",herm=false)
	test_mirror_block(V,p;name="V",herm=true)
	test_mirror_block(V,p;name="V",herm=false)
	test_mirror_block(p.𝔸1,p;name="A1",herm=true)
	test_mirror_block(p.𝔸1,p;name="A1",herm=false)
	test_mirror_block(p.𝔸2,p;name="A2",herm=true)
	test_mirror_block(p.𝔸2,p;name="A2",herm=false)
	test_mirror_block(p.Σ,p;name="Σ",herm=true)
	test_mirror_block(p.Σ,p;name="Σ",herm=false)

	# R
	px("\nTests R symmetry")
	test_R_block(T,p;name="T")
	test_R_block(W,p;name="W")
	test_R_block(V,p;name="V")
	test_R_magnetic_block(p.𝔸1,p.𝔸2,p;name="A")
	test_R_block(p.Σ,p;name="Σ")

	# Equalities inside blocks
	px("\nTests equality inside blocks")
	test_equality_all_blocks(T,p;name="T")
	test_equality_all_blocks(W,p;name="W")
	test_equality_all_blocks(V,p;name="V")
	test_equality_all_blocks(p.Σ,p;name="Σ")
	test_equality_all_blocks(p.𝔸1,p;name="A1")
	test_equality_all_blocks(p.𝔸2,p;name="A2")
	# px("Equality blocks V and V_moins ",relative_distance_blocks(V,V_moins))

	# Hermitianity
	px("\nTests hermitianity")
	test_block_hermitianity(W,p;name="W")

	if produce_plots
		# Plots in reduced coordinates
		plot_block_reduced(W,p;title="W")
		plot_block_reduced(V,p;title="V")
		plot_block_reduced(p.Wplus,p;title="W_plus")
		plot_block_reduced(p.Wmoins,p;title="W_moins")
		plot_block_reduced(p.Σ,p;title="Σ")
		plot_block_reduced(p.𝔸1,p;title="A1")
		plot_block_reduced(p.𝔸2,p;title="A2")

		# Plots in cartesian coordinates
		plot_block_cart(T,p;title="T")
		plot_block_cart(p.Wplus,p;title="W_plus")
		plot_block_cart(p.Wmoins,p;title="W_moins")
		plot_block_cart(p.𝕍,p;title="V")
		plot_block_cart(p.𝕍_V,p;title="V_V")
		plot_block_cart(p.𝕍_Vint,p;title="V_Vint")
		plot_block_cart(p.Σ,p;title="Σ")
		plot_magnetic_block_cart(p.𝔸1,p.𝔸2,p;title="A") 
		# plot_magnetic_block_cart(p.𝔹1,p.𝔹2,p;title="B") 
		# plot_block_cart(p.𝔹1,p;title="B1")
		# plot_block_cart(p.𝔹2,p;title="B2")
		if take_Vint_into_account
			plot_block_cart(p.W,p;title="W")
			plot_block_cart(p.𝕍_Vint,p;title="V_Vint")
			plot_block_cart(p.𝕍,p;title="V")
		end
	end
end

#################### Third step : compute the bands diagram

function explore_band_structure_BM()
	p = Basis()
	p.N = 8
	p.a = 4.66
	p.l = 6 # number of eigenvalues we compute
	init_basis(p)
	α = 0.0
	p.resolution_bands = 5

	Vmat0 = V_offdiag_matrix(T_BM_four(p.N,p.a,1,1),p)

	p.folder_plots_bands = "bands_BM"
	p.energy_center = 0
	p.energy_scale = 2
	for β in vcat((0:0.05:6),(1.2:0.001:1.25))
		print(" ",β)
		Vmat = weights_off_diag_matrix(Vmat0,α,β,p)

		Hv = p.H0 + Vmat
		# px("mass V ",sum(abs.(Vmat)))
		# test_hermitianity(Hv)
		s = string(β,"00000000000")
		plot_band_structure(Hv,s[1:min(6,length(s))],p)
	end
	p
end

function explore_band_structure_Heff()
	N = 8; Nz = 27
	sc = 1.5*1e3

	# Imports u1, u2, V, Vint and computes the effective potentials
	EffV = import_and_computes(N,Nz)

	p = Basis()
	p.N = N; p.a = EffV.a
	p.l = 12 # number of eigenvalues we compute
	init_basis(p)

	######## Base Hamiltonian
	# Mass matrix
	SΣ = V_offdiag_matrix(lin2mat(EffV.Σ),p)/sc
	S = Hermitian(I + 1*SΣ)
	p.ISΣ = Hermitian(inv(sqrt(S)))
	# test_hermitianity(S,"S"); test_part_hole_sym_matrix(S,p,"S")
	
	# On-diagonal potential
	V = V_ondiag_matrix(lin2mat(EffV.𝕍_V),p)/sc # WHY THIS IS NOT CONSTANT AS IN THE COMPUTATION ????
	
	# Off-diagonal magnetic operator
	A∇ = A_offdiag_matrix(lin2mat(EffV.𝔸1),lin2mat(EffV.𝔸2),p)/sc

	# Off-diagonal potential
	W = V_offdiag_matrix(lin2mat(EffV.Wplus),p)/sc

	# Other parameters
	p.solver=="Exact"

	p.folder_plots_bands = "bands_eff_avec_V_A_Sigma_alpha_egal_1"
	p.energy_center = -0.5
	p.energy_scale = 2
	p.resolution_bands = 5

	method = "natural" # ∈ ["weight","natural"]
	if method=="natural"
		for θ in (0.005:0.001:0.04) # 1° × 2π/360 = 0.017 rad
			cθ = cos(θ/2); εθ = sin(θ/2)
			Hv = p.ISΣ*( cθ*(p.H0 + A∇) + (1/εθ)*(V+0*W) )*p.ISΣ
			print(" ",θ)
			# px("mass W ",sum(abs.(W)))
			# test_hermitianity(Hv); test_part_hole_sym_matrix(W,p,"W")
			s = string(θ,"00000000000")
			title = s[1:min(6,length(s))]
			title = string(θ)
			plot_band_structure(Hv,title,p)
		end
	else
		H1 = p.H0 + 0*V + A∇
		α = 1.0
		for β in (0:1:2)
			print(" ",β)
			W_weighted = weights_off_diag_matrix(W0,α,β,p)
			# px("mass W ",sum(abs.(W)))
			Hv = p.ISΣ*(H1 + W_weighted)*p.ISΣ
			# test_hermitianity(Hv); test_part_hole_sym_matrix(W,p,"W")
			s = string(β,"00000000000")
			plot_band_structure(Hv,s[1:min(6,length(s))],p)
		end
	end
end

# computes_and_plots_effective_potentials()
explore_band_structure_Heff()
# explore_band_structure_BM()
#

#### Todo
# chemin prop sur diag bandes
# Wplus =? Wmoins
# régler pb W trop grand
# cube Fourier pour plus de symétrie

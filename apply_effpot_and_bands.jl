include("band_diagrams_bm_like.jl")
using DelimitedFiles

#################### Second step : compute the effective potentials 𝕍, 𝕎, 𝔸, etc. Very rapid

function computes_and_plots_effective_potentials()
	# Parameters
	# N = 8; Nz = 27
	# N = 9;  Nz = 32 # has Vint
	# N = 9;  Nz = 36
	N = 12; Nz = 40
	# N = 12; Nz = 45
	N = 15; Nz = 54
	# N = 15; Nz = 60
	# N = 20; Nz = 72
	# N = 24; Nz = 90
	# N = 24; Nz = 96
	# N = 25; Nz = 108 # ecut 40|Kd|^2
	# N = 30; Nz = 120 # ecut 50|Kd|^2 which blocks because of a bug on DFTK
	N = 30; Nz = 125 # ecut 55|Kd|^2
	# N = 32; Nz = 135 # ecut 50
	# N = 40; Nz = 160
	N = 40; Nz = 180 # <-- Ecut = 100 |Kd|^2
	# N = 45; Nz = 180 # Ecut = 120|Kd|^2
	# N = 45; Nz = 192
	# N = 48; Nz = 200

	px("N ",N,", Nz ",Nz)
	p = EffPotentials()

	p.plots_cutoff = 3
	p.plots_res = 100
	p.plots_n_motifs = 6
	produce_plots = false
	p.compute_Vint = true
	p.plot_for_article = true

	# Imports untwisted quantities
	import_u1_u2_V_φ(N,Nz,p)
	import_Vint(p)
	init_EffPot(p)
	# px("Test norm ",norms3d(p.u1_dir,p,false)," and in Fourier ",norms3d(p.u1_f,p))


	px("SQRT Cell area ",sqrt(p.cell_area))

	build_blocks_potentials(p) # computes Wplus, 𝕍_V and Σ
	compare_to_BM_infos(p.𝕍_V,p,"V_V")
	optimize_gauge_and_create_T_BM_with_α(true,p) # optimizes on α only, not on the global phasis, which was already well-chosen before at the graphene.jl level
	T_BM = build_BM(5,5,p)
	compare_to_BM_infos(T_BM,p,"T_BM")
	px("Compare")
	compare_blocks(T_BM,p.𝕍_V,p)

	build_blocks_potentials(p) # computes Wplus, 𝕍_V and Σ
	compare_to_BM_infos(p.𝕍_V,p,"V_V")

	# plot_block_reduced(p.T_BM,p;title="T")
	# plot_block_reduced(p.𝕍_V,p;title="V_V")

	(wAA,wAB) = hartree_to_ev .*wAA_wAB(p)
	px("wAA = ",wAA," eV, wAB = ",wAB," eV")
	px("IL FAUT AUSSI PRENDRE EN COMPTE VINT !")

	# plot_block_reduced(p.T_BM,p;title="T")
	p.add_non_local_W = true
	px("Distance between Σ and T_BM ",relative_distance_blocks(p.Σ,p.T_BM)) # MAYBE NOT PRECISE !
	px("Distance between V_V and T_BM ",relative_distance_blocks(p.𝕍_V,p.T_BM))

	# Compares functions of T and 𝕍
	px("Comparision to BM")
	compare_to_BM_infos(p.𝕍_V,p,"V_V") # normal that individual blocks distances are half the total distance because there are two blocks each time
	compare_to_BM_infos(p.Σ,p,"Σ")
	build_block_𝔸(p) # computes 𝔸

	px("V_{11}(x-(1/3)(a1-a2)) = V_{12}(x) ",distance(translation_interpolation(p.𝕍_V[1], [1/3,-1/3],p),p.𝕍_V[2]))
	px("V_{11}(x+(1/3)(a1-a2)) = V_{12}(x) ",distance(translation_interpolation(p.𝕍_V[1],-[1/3,-1/3],p),p.𝕍_V[2]))

	# plot_block_cart(p.Σ,p;title="Σ")
	# plot_block_cart(p.𝕍_V,p;title="V_V")
	# plot_block_cart(p.T_BM,p;title="T")
	# p.𝕍 = app_block(J_four,p.𝕍,p) # rotates T of J and rescales space of sqrt(3)
	# test_equality_all_blocks(p.Wplus_tot,p;name="W")
	# plot_block_cart(p.W_non_local_plus,p;title="W_nl_plus")

	# testit(p)


	# Mean W
	means_W_V = mean_block(p.W_V_plus,p)
	means_W_V_minus = mean_block(p.W_V_minus,p)
	@assert distance(means_W_V_minus,means_W_V_minus)<1e-4
	W_V_without_mean = add_cst_block(p.W_V_plus,-sqrt(p.cell_area)*means_W_V,p)
	px("Mean W_V_plus (in Fourier) :")
	display(means_W_V)
	px("\nW_Vint matrix")
	display(p.W_Vint_matrix)
	px("Mean W_plus in meV :")
	display(mean_block(p.W_V_minus,p)*1e3*hartree_to_ev)

	# Plots for article
	if p.plot_for_article
		# plot_block_reduced(p.𝕍_V,p;title="V_V")
		# plot_block_reduced(p.𝕍,p;title="V")
		# plot_block_article(p.𝕍,p;title="V",k_red_shift=-p.m_q1)
		# plot_block_article(p.T_BM,p;title="T",k_red_shift=-p.m_q1)
		# plot_block_article(W_V_without_mean,p;title="W_plus_without_mean")
		# plot_block_article(p.𝔸1,p;title="A",other_block=p.𝔸2,k_red_shift=-p.m_q1)
		# if p.compute_Vint plot_block_article(p.𝕍_Vint,p;title="V_Vint",k_red_shift=-p.m_q1) end
		plot_block_article(p.Σ,p;title="Σ",k_red_shift=-p.m_q1,meV=false)
		# plot_block_article(p.W_non_local_plus,p;title="W_nl_plus",k_red_shift=-p.m_q1)
	end


	####################### Symmetries
	# Tests z parity
	# test_z_parity(p.u1_dir,-1,p;name="u1")
	# test_z_parity(p.u2_dir,-1,p;name="u2")
	# test_z_parity(p.v_dir,1,p; name="Vks")

	# Particle-hole
	px("\nTests particle-hole symmetry")
	test_particle_hole_block(p.T_BM,p;name="T")
	test_particle_hole_block(p.𝕍,p;name="V")
	test_particle_hole_block_W(p.W_V_plus,p.W_V_minus,p;name="W_V")
	test_particle_hole_block_W(p.W_non_local_plus,p.W_non_local_minus,p;name="Wnl")
	test_particle_hole_block(p.Σ,p;name="Σ")
	test_particle_hole_block(p.𝔸1,p;name="A1")
	test_particle_hole_block(p.𝔸2,p;name="A2")

	# Parity-time
	px("\nTests PT symmetry")
	test_PT_block(p.T_BM,p;name="T")
	test_PT_block(p.Wplus_tot,p;name="W+")
	test_PT_block(p.Wminus_tot,p;name="W-")
	test_PT_block(p.𝕍,p;name="V")
	test_PT_block(p.𝔸1,p;name="A1")
	test_PT_block(p.𝔸2,p;name="A2")
	test_PT_block(p.Σ,p;name="Σ")
	test_PT_block(p.W_non_local_plus,p;name="Wnl+")
	test_PT_block(p.W_non_local_minus,p;name="Wnl-")

	# Special symmetry of W
	test_sym_Wplus_Wminus(p)

	# Mirror
	px("\nTests mirror symmetry")
	test_mirror_block(p.T_BM,p;name="T",herm=true)
	test_mirror_block(p.Wplus_tot,p;name="W",herm=true)
	test_mirror_block(p.Wplus_tot,p;name="W",herm=false)
	test_mirror_block(p.𝕍,p;name="V",herm=true)
	test_mirror_block(p.𝕍,p;name="V",herm=false)
	test_mirror_block(p.𝔸1,p;name="A1",herm=true)
	test_mirror_block(p.𝔸1,p;name="A1",herm=false)
	test_mirror_block(p.𝔸2,p;name="A2",herm=true)
	test_mirror_block(p.𝔸2,p;name="A2",herm=false)
	test_mirror_block(p.Σ,p;name="Σ",herm=true)
	test_mirror_block(p.Σ,p;name="Σ",herm=false)
	test_mirror_block(p.W_non_local_plus,p;name="Wnl+",herm=false)
	test_mirror_block(p.W_non_local_minus,p;name="Wnl-",herm=false)

	px("Compare W+ and W- ",distance(p.W_V_plus,p.W_V_minus))

	# R
	px("\nTests R symmetry")
	test_R_block(p.T_BM,p;name="T")
	test_R_block(p.Wplus_tot,p;name="W")
	test_R_block(p.𝕍,p;name="V")
	test_R_magnetic_block(p.𝔸1,p.𝔸2,p;name="A")
	test_R_block(p.Σ,p;name="Σ")
	test_R_block(p.W_non_local_plus,p;name="Wnl+")
	test_R_block(p.W_non_local_minus,p;name="Wnl-")

	# Equalities inside blocks
	# px("\nTests equality inside blocks")
	# test_equality_all_blocks(p.T_BM,p;name="T")
	# test_equality_all_blocks(p.Wplus_tot,p;name="W")
	# test_equality_all_blocks(p.𝕍,p;name="V")
	# test_equality_all_blocks(p.Σ,p;name="Σ")
	# test_equality_all_blocks(p.𝔸1,p;name="A1")
	# test_equality_all_blocks(p.𝔸2,p;name="A2")
	# px("Equality blocks V and V_minus ",relative_distance_blocks(V,V_minus))

	# Hermitianity
	px("\nTests hermitianity")
	test_block_hermitianity(p.Wplus_tot,p;name="W")
	px("\n")

	if produce_plots
		# Plots in reduced coordinates
		plot_block_reduced(p.T_BM,p;title="T")
		plot_block_reduced(p.Wplus_tot,p;title="W")
		plot_block_reduced(p.𝕍,p;title="V")
		plot_block_reduced(p.Σ,p;title="Σ")
		plot_block_reduced(p.𝔸1,p;title="A1")
		plot_block_reduced(p.𝔸2,p;title="A2")

		# Plots in cartesian coordinates
		plot_block_cart(p.T_BM,p;title="T",article=true)

		# W
		plot_block_cart(p.W_V_plus,p;title="W_V_plus",article=true)
		plot_block_cart(p.W_V_minus,p;title="W_V_minus")

		plot_block_cart(p.W_non_local_plus,p;title="W_nl_plus",article=true)
		plot_block_cart(p.W_non_local_minus,p;title="W_nl_moins")

		plot_block_cart(p.Wplus_tot,p;title="W_plus_tot")

		# V
		plot_block_cart(p.𝕍,p;title="V")
		plot_block_cart(p.𝕍_V,p;title="V_V",article=true)
		plot_block_cart(p.𝕍_Vint,p;title="V_Vint",article=true)

		# Σ and A
		plot_block_cart(p.Σ,p;title="Σ",article=true)
		plot_magnetic_block_cart(p.𝔸1,p.𝔸2,p;title="A",article=true)
		# plot_magnetic_block_cart(p.𝔹1,p.𝔹2,p;title="B") 
		# plot_block_cart(p.𝔹1,p;title="B1")
		# plot_block_cart(p.𝔹2,p;title="B2")
	end
	p
end

#################### Third step : compute the bands diagram

# Reproduire TKV :
# p.a1_star = [ sqrt(3)/2; 1/2]
# p.a2_star = [-sqrt(3)/2; 1/2]
# p.a doesn't matter it is not used
# T = V_offdiag_matrix(build_BM(α,β,p;scale=false),p)*sqrt(p.cell_area)
# for β in [0.586]

function explore_band_structure_BM()
	p = Basis()
	p.N = 8
	# @assert mod(p.N,2)==1 # for more symmetries
	p.a_micro = 8π/sqrt(3)
	p.dim = 2
	p.l = 11 # number of eigenvalues we compute
	init_basis(p)
	α = 0 # anti-chiral / AA stacking weight
	p.resolution_bands = 6
	p.energy_unit_plots = "Hartree"
	p.folder_plots_bands = "bands_BM"
	p.energy_center = 0
	p.energy_scale = 0.85
	p.solver = "Exact"
	mult_by_vF = true
	p.coef_derivations = 1


	no = norm(p.q2)
	update_a(p.q2/no,p.q3/no,p)

	K1 = [1/3,2/3]
	K2 = [2/3,1/3]
	K_reds = [K2,K1]

	Γ = [0,0.0]
	K = p.K_red
	px("Kred ",p.K_red)
	M = [0,1/2]

	Klist = [Γ,K,M]; Klist_names = ["Γ","K","M"]
	A = K2 # K'
	B = K1 # K
	C = 2*K1-K2 # Γ1
	M = C/2
	D = [0,0]

	Klist = [A,B,C,M,D]
	# Klist = [Klist[i]-p.K_red for i=1:length(Klist)]

	Klist_names = ["A","B","C","M","D"]
	# Klist = [Γ,K,M]; Klist_names = ["Γ","K","M"]

	Kf = [k -> Dirac_k(k-K_reds[i],p;coef_∇=0) for i=1:2]
	# Kf(k) = Dirac_k(k,p;coef_∇=0)

	H0 = Dirac_k([0.0,0.0],p)
	# for β in [0.586]
	for β in vcat([0])
	# for β in vcat([0,0.586])
	# for β in vcat([0.586],(0:0.05:0.4))
		print(" ",β)
		TBM = build_BM(α,β,p)
		TBM3 = rescale_A_block(TBM,p;shift=true)
		T = V_offdiag_matrix(TBM3,p)*sqrt(p.cell_area)

		σs = [spectrum_on_a_path(H0.+T,Kf[i],Klist,p) for i=1:2]
		pl = plot_band_diagram(σs,Klist,Klist_names,p)
		save_diagram(pl,β,p)
	end
	p
end

function explore_band_structure_BM2()
	p = Basis()
	p.N = 7
	@assert mod(p.N,2)==1 # for more symmetries
	p.a = 1 # decreasing a makes band energies increase
	p.l = 11 # number of eigenvalues we compute
	init_basis(p)
	α = 1 # anti-chiral / AA stacking weight
	p.resolution_bands = 4
	p.energy_unit_plots = "Hartree"
	p.folder_plots_bands = "bands_BM"
	p.energy_center = 0
	p.energy_scale = 0.2
	p.solver = "Exact"
	mult_by_vF = true
	p.coef_derivations = 1
	# TKV EST UN REPLIEMENT DES BANDES !!!!!!!!!!!!!!!!!!
	# TKV EST UN REPLIEMENT DES BANDES !!!!!!!!!!!!!!!!!!
	# TKV EST UN REPLIEMENT DES BANDES !!!!!!!!!!!!!!!!!!

	# T = V_offdiag_matrix(build_BM(0,1,p;scale=false),p)
	# plot_heatmap(imag.(T).+real.(T),"T_BM",p)
	# plot_heatmap(real.(Dirac_k([0,0],p)),"free_dirac",p)

	# 1° × 2π/360 = 0.017 rad
	# for β in vcat((0.1:0.1:1)) # chiral / AB stacking weight
	# for β in (0:0.05:1.2)
	# for β in vcat((0:0.05:1.2),(0.85:0.01:1)) # chiral / AB stacking weight
	
	Γ = [0,0.0]
	K = p.K_red
	M = [0,1/2]
	Klist = [K,Γ,M]; Klist_names = ["K","Γ","M"]
	valleys = [1,-1]
	K_reds = [v*[-1/3,1/3] for v in valleys]
	Kf = [k -> Dirac_k(k.-K_reds[i],p;coef_∇=0,valley=valleys[i]) for i=1:2]

	H0 = Dirac_k([0.0,0.0],p)
	for β in [0.586]
		print(" ",β)
		T = V_offdiag_matrix(build_BM(α,β,p;scale=false),p)*sqrt(p.cell_area)
		σs = [spectrum_on_a_path(H0.+T,Kf[i],Klist,p) for i=1:2]
		pl = plot_band_diagram(σs,Klist,Klist_names,p)#;K_relative=p.K_red)
		save_diagram(pl,β,p)
	end
	p
end

function explore_free_graphene_bands()
	p = Basis()
	p.N = 8
	p.a = 4.66
	# p.a = 4.66
	p.l = p.N^2-1 # number of eigenvalues we compute
	p.double_dirac = false
	init_basis(p)
	p.resolution_bands = 11
	p.energy_unit_plots = "Hartree"
	p.folder_plots_bands = "bands_free_graphene"
	p.energy_scale = 2
	p.energy_center = 0
	p.solver = "Exact"
	mult_by_vF = true
	p.coef_derivations = 1

	Kdep(k_red) = free_Dirac_k_monolayer(k_red,p)
	# Kdep(k_red) = 0.05*free_Schro_k_monolayer(k_red,p)
	# K-independent part
	Hv = 0*Kdep([0,0.0])
	plot_band_structure(Hv,Kdep,"free",p)
	p
end

# CHOQUANT : Σ = T !!!!!!!!!!!?!!!!
function explore_band_structure_Heff()
	N = 8; Nz = 27
	N = 32; Nz = 135

	# Imports u1, u2, V, Vint, v_fermi and computes the effective potentials

	compute_Vint = false
	EffV = import_and_computes(N,Nz,compute_Vint)
	p.add_non_local_W = true

	p = Basis()
	p.N = N; p.a = EffV.a
	p.l = 12 # number of eigenvalues we compute
	p.coef_derivations = 1
	init_basis(p)

	######## Base Hamiltonian
	# Mass matrix
	SΣ = V_offdiag_matrix(EffV.Σ,p)
	S = Hermitian(I + 1*SΣ)
	p.ISΣ = Hermitian(inv(sqrt(S)))
	# test_hermitianity(S,"S"); test_part_hole_sym_matrix(S,p,"S")
	
	# On-diagonal potential
	W = V_ondiag_matrix(EffV.Wplus_tot,EffV.Wminus_tot,p) # WHY THIS IS NOT CONSTANT AS IN THE COMPUTATION ????
	
	# Off-diagonal potential
	V = V_offdiag_matrix(EffV.𝕍,p)

	# Other parameters
	p.solver="Exact"

	p.folder_plots_bands = "bands_eff"
	p.energy_center = -0.5
	p.energy_scale = 10
	p.resolution_bands = 10
	p.energy_unit_plots = "eV"

	method = "natural" # ∈ ["weight","natural"]
	if method=="natural"
		# for θ in (0.01:0.01:0.3) # 1° × 2π/360 = 0.017 rad
		for θ_degres in (0.1:0.2:3)
		# for θ in (0.0001:0.0001:0.001)
			θ = 0.017*θ_degres
			print(" ",θ)
			# p.a = sqrt(3)*4.66/(2*sin(θ/2))
			reload_a(p)
			cθ = cos(θ/2); εθ = sin(θ/2)
			# If needed to accelerate : compute all the operators for all k, then multiply by each constant depending on θ. Ici on forme plein de fois des operateurs HkV alors qu'on peut l'éviter

			# K-dependent part
			function Kdep(k_red)
				# Off-diagonal magnetic operator
				A∇ = A_offdiag_matrix(EffV.𝔸1,EffV.𝔸2,k_red,p)
				JA∇ = A_offdiag_matrix(EffV.J𝔸1,EffV.J𝔸2,k_red,p)
				ΣmΔ = VΔ_offdiag_matrix(EffV.Σ,k_red,p)
				Δ = mΔ(k_red,p)
				EffV.v_fermi*p.ISΣ*(cθ*(Dirac_k(k_red,p) +A∇) + εθ*(0.5*(Δ + ΣmΔ) - JA∇ + J_Dirac_k(k_red,p)))*p.ISΣ
			end

			# K-independent part
			Hv = p.ISΣ*( (1/εθ)*(V+ 0*W) )*p.ISΣ

			# px("mass W ",sum(abs.(W)))
			# test_hermitianity(Hv)#; test_part_hole_sym_matrix(W,p,"W")
			s = string(θ_degres,"00000000000")
			title = s[1:min(6,length(s))]
			plot_band_structure(Hv,Kdep,title,p)
		end
	else
		H1 = p.H0 + V + A∇
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

# p = computes_and_plots_effective_potentials()
# explore_band_structure_Heff()
explore_band_structure_BM()
# explore_free_graphene_bands()
nothing

#### Todo
# FAIRE GRAPH AVEC WAB ET SUIVANT EN FONCTION DE d, norm(Σ,∇Σ,W,V,V-T (qui aille plus vite vers 0 que V))
#  Donner les coefs suivants de wAA et wAB
# FAIRE GRAPH AVEC BILAYER, a_M et qj^*, et TBM
# Donner wAA et wAB avec les 6 modes de Fourier. Pq real et imag sont pas invariants sous 2π/3 ?
# Régler pb de phase pour W^nl
# DANS TKV IL Y A LES DEUX VALLEES !!!

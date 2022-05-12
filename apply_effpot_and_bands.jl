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
	# N = 15; Nz = 54
	# N = 15; Nz = 60
	# N = 20; Nz = 72
	# N = 24; Nz = 90
	# N = 24; Nz = 96
	# N = 25; Nz = 108 # ecut 40|Kd|^2
	# N = 30; Nz = 120 # ecut 50|Kd|^2 which blocks because of a bug on DFTK
	# N = 30; Nz = 125 # ecut 55|Kd|^2
	# N = 32; Nz = 135 # ecut 50
	# N = 40; Nz = 160
	N = 40; Nz = 180 # <-- Ecut = 100 |kD|^2
	# N = 45; Nz = 180 # ecut 120|Kd|^2
	# N = 45; Nz = 192
	# N = 48; Nz = 200

	px("N ",N,", Nz ",Nz)
	p = EffPotentials()

	p.plots_cutoff = 7
	p.plots_res = 40
	p.plots_n_motifs = 6
	produce_plots = false
	p.compute_Vint = false
	p.plot_for_article = true

	# Imports untwisted quantities
	import_u1_u2_V_φ(N,Nz,p)
	import_Vint(p)
	init_EffPot(p)
	# px("Test norm ",norms3d(p.u1_dir,p,false)," and in Fourier ",norms3d(p.u1_f,p))

	(wAA,wAB) = hartree_to_ev .*wAA_wAB(p)
	px("wAA = ",wAA," eV, wAB = ",wAB," eV")
	px("IL FAUT AUSSI PRENDRE EN COMPTE VINT !")

	px("Cell area ",sqrt(p.cell_area))

	build_blocks_potentials(p) # computes Wplus, 𝕍_V and Σ
	compare_to_BM_infos(p.𝕍_V,p,"V_V")
	optimize_gauge_and_create_T_BM_with_α(true,p) # optimizes on α only, not on the global phasis, which was already well-chosen before at the graphene.jl level

	build_blocks_potentials(p) # computes Wplus, 𝕍_V and Σ
	compare_to_BM_infos(p.𝕍_V,p,"V_V")

	plot_block_reduced(p.T_BM,p;title="T")

	# plot_block_reduced(p.T_BM,p;title="T")
	p.add_non_local_W = true
	px("Distance between Σ and T_BM ",relative_distance_blocks(p.Σ,p.T_BM)) # MAYBE NOT PRECISE !
	px("Distance between V_V and T_BM ",relative_distance_blocks(p.𝕍_V,p.T_BM))

	# Compares functions of T and 𝕍
	px("Comparision to BM")
	compare_to_BM_infos(p.𝕍_V,p,"V_V") # normal that individual blocks distances are half the total distance because there are two blocks each time
	compare_to_BM_infos(p.Σ,p,"Σ")
	build_block_𝔸(p) # computes 𝔸


	means_W_V = [p.W_V_plus[i][1,1] for i=1:4]/sqrt(p.cell_area)
	W_V_without_mean = add_cst_block(p.W_V_plus,-1*means_W_V,p)
	px("Mean W_V_plus (in Fourier) :")
	display(means_W_V)
	# plot_block_article(W_V_without_mean,p;title="W_V_plus_without_mean")

	# plot_block_cart(p.Σ,p;title="Σ")
	# plot_block_cart(p.𝕍_V,p;title="V_V")
	# plot_block_cart(p.T_BM,p;title="T")
	# p.𝕍 = app_block(J_four,p.𝕍,p) # rotates T of J and rescales space of sqrt(3)
	# test_equality_all_blocks(p.Wplus_tot,p;name="W")
	# plot_block_cart(p.W_non_local_plus,p;title="W_nl_plus")

	# testit(p)

	px("\nW_Vint matrix")
	display(p.W_Vint_matrix)

	# Plots for article
	if p.plot_for_article
		# plot_block_article(p.T_BM,p;title="T")
		plot_block_article(p.𝕍_V,p;title="V_V",k_red_shift=[1/3,1/3])
		# if p.compute_Vint plot_block_article(p.𝕍_Vint,p;title="V_Vint") end
		# plot_block_article(p.Σ,p;title="Σ")
		# plot_block_article(p.W_non_local_plus,p;title="W_nl_plus")
		# plot_block_article(p.𝔸1*1/2,p;title="A",other_block=p.𝔸2*1/2)
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
	p.N = 7
	@assert mod(p.N,2)==1 # for more symmetries
	p.a = 1 # decreasing a makes band energies increase
	p.l = 11 # number of eigenvalues we compute
	init_basis(p)
	α = 1 # anti-chiral / AA stacking weight
	p.resolution_bands = 20
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
	# TKV EST UN REPLIEMENT DES BANDES !!!!!!!!!!!!!!!!!!
	# TKV EST UN REPLIEMENT DES BANDES !!!!!!!!!!!!!!!!!!
	# TKV EST UN REPLIEMENT DES BANDES !!!!!!!!!!!!!!!!!!
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

computes_and_plots_effective_potentials()
# explore_band_structure_Heff()
# explore_band_structure_BM()
# explore_free_graphene_bands()
nothing

#### Todo
# voir si \cA peut pas etre sous la forme sum_123 A_j e^{ix qj}
#
# ajouter effet du terme non local
# cube Fourier pour plus de symétrie
# Ht_a ≂̸ t_a H comme dit par Watson, regarder son papier sur l'existence des magic angles
# régler le pb du scaling -3/2 JX
#
# Régler pb de la convergence des pot effectifs quand N est grand
# Reproduire diagramme de bandes de Tarnopolsky
#
# Does non local φ depends on the gauge we choose on wavefunctions ? in this case we should adapt it when we fix the gauge
#
# VERIFIER SYMMETRIEs

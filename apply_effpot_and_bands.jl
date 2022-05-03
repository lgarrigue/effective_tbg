include("band_diagrams_bm_like.jl")

#################### Second step : compute the effective potentials 𝕍, 𝕎, 𝔸, etc. Very rapid

function computes_and_plots_effective_potentials()
	# Parameters
	# N = 8; Nz = 27
	# N = 9;  Nz = 36
	# N = 12; Nz = 45
	# N = 15; Nz = 60
	# N = 20; Nz = 72
	# N = 24; Nz = 90
	# N = 24; Nz = 96
	N = 32; Nz = 135 # ecut 50
	# N = 40; Nz = 160
	# N = 45; Nz = 192
	# N = 48; Nz = 200

	px("N ",N,", Nz ",Nz)
	p = EffPotentials()
	p.plots_cutoff = 7
	p.plots_res = 60
	p.plots_n_motifs = 6
	produce_plots = false
	p.compute_Vint = false

	# Initializations
	import_u1_u2_V_φ(N,Nz,p)
	import_Vint(p)
	init_EffPot(p)
	# px("Test norm ",norms3d(p.u1_dir,p,false)," and in Fourier ",norms3d(p.u1_f,p))

	optimize_gauge_and_create_T_BM_with_θ_α(false,p)
	optimize_gauge_and_create_T_BM_with_α(true,p)

	plot_block_reduced(p.T_BM,p;title="T")
	p.add_non_local_W = true
	build_blocks_potentials(p) # computes Wplus, 𝕍_V and Σ
	px("Distance between Σ and T_BM ",relative_distance_blocks(p.Σ,p.T_BM)) # MAYBE NOT PRECISE !
	px("Distance between V_V and T_BM ",relative_distance_blocks(p.𝕍_V,p.T_BM))

	# Compares functions of T and 𝕍
	px("Comparision to BM")
	compare_to_BM_infos(p.𝕍_V,p,"V_V") # normal that individual blocks distances are half the total distance because there are two blocks each time
	compare_to_BM_infos(p.Σ,p,"Σ")

	build_block_𝔸(p) # computes 𝔸
	# plot_block_cart(p.𝕍_V,p;title="V_V")
	# plot_block_cart(p.Wplus,p;title="W_plus")
	# plot_block_cart(p.Σ,p;title="Σ")
	# plot_block_reduced(p.𝕍,p;title="V")
	# plot_block_cart(p.T_BM,p;title="T")
	# p.𝕍 = app_block(J_four,p.𝕍,p) # rotates T of J and rescales space of sqrt(3)
	# test_equality_all_blocks(p.Wplus_tot,p;name="W")
	plot_block_cart(p.W_non_local_plus,p;title="W_nl_plus")

	# testit(p)

	px("\nW_Vint matrix")
	display(p.W_Vint_matrix)

	test_z_parity(p.u1_dir,-1,p;name="u1")
	test_z_parity(p.u2_dir,-1,p;name="u2")
	test_z_parity(p.v_dir,1,p;name="Vks")

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
		plot_block_cart(p.T_BM,p;title="T")
		plot_block_cart(p.Wplus,p;title="W_plus")
		plot_block_cart(p.Wplus_tot,p;title="W_plus_tot")
		plot_block_cart(p.Wminus,p;title="W_minus")
		plot_block_cart(p.𝕍,p;title="V")
		plot_block_cart(p.𝕍_V,p;title="V_V")
		plot_block_cart(p.𝕍_Vint,p;title="V_Vint")
		plot_block_cart(p.Σ,p;title="Σ")
		plot_block_cart(p.W_non_local_plus,p;title="W_nl_plus")
		plot_block_cart(p.W_non_local_moins,p;title="W_nl_moins")
		plot_magnetic_block_cart(p.𝔸1,p.𝔸2,p;title="A") 
		# plot_magnetic_block_cart(p.𝔹1,p.𝔹2,p;title="B") 
		# plot_block_cart(p.𝔹1,p;title="B1")
		# plot_block_cart(p.𝔹2,p;title="B2")
	end
end

#################### Third step : compute the bands diagram

function explore_band_structure_BM()
	p = Basis()
	p.N = 7
	p.a = 4. # decreasing a makes band energies increase
	p.l = 8 # number of eigenvalues we compute
	init_basis(p)
	α = 0.0 # anti-chiral / AA stacking weight
	p.resolution_bands = 4
	p.energy_unit_plots = "Hartree"
	p.folder_plots_bands = "bands_BM"
	p.energy_center = 0
	p.energy_scale = 1.5
	p.solver = "Exact"
	mult_by_vF = true
	p.coef_derivations = 1
	# 1° × 2π/360 = 0.017 rad
	# for β in vcat((0.1:0.1:1)) # chiral / AB stacking weight
	# for β in (0:0.05:1.2)
	# for β in vcat((0:0.05:1.2),(0.85:0.01:1)) # chiral / AB stacking weight
	for β in (0:0.1:7)
		print(" ",β)
		T = V_offdiag_matrix(build_BM(α,β,p;scale=false),p)
		Kdep(k_red) = Dirac_k(k_red,p)
		plot_band_structure(T,Kdep,β,p)
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

# computes_and_plots_effective_potentials()
# explore_band_structure_Heff()
explore_band_structure_BM()
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

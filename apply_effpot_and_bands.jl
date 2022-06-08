include("band_diagrams_bm_like.jl")
using DelimitedFiles, CairoMakie, LaTeXStrings

using AbstractPlotting.MakieLayout
using AbstractPlotting
using AbstractPlotting: px

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
	N = 25; Nz = 160 # ecut
	# N = 25; Nz = 160 # ecut
	# N = 25; Nz = 216 # ecut
	# N = 25; Nz = 270 # ecut
	# N = 25; Nz = 320 # ecut
	# N = 25; Nz = 432 # ecut
	# N = 25; Nz = 625 # ecut
	# N = 27; Nz = 180 # ecut
	N = 27; Nz = 600 # <----------------------------------------
	# N = 30; Nz = 120 # ecut 50|Kd|^2 which blocks because of a bug on DFTK
	# N = 30; Nz = 125 # ecut 55|Kd|^2
	# N = 30; Nz = 243 # ecut 55|Kd|^2
	# N = 32; Nz = 135 # ecut 50
	# N = 40; Nz = 160
	# N = 40; Nz = 180 # <-- Ecut = 100 |Kd|^2
	# N = 32; Nz = 256 # <-- L=40, Ecut = 60 |Kd|^2
	# N = 32; Nz = 225 # <-- L=35, Ecut = 60 |Kd|^2
	# N = 32; Nz = 192 # <-- L=30, Ecut = 60 |Kd|^2
	# N = 32; Nz = 320 # <-- L=50, Ecut = 60 |Kd|^2
	# N = 45; Nz = 180 # Ecut = 120|Kd|^2
	# N = 45; Nz = 192
	# N = 48; Nz = 200

	compute_Vint = true
	interlayer_distance = 6.45
	p = import_and_computes(N,Nz,compute_Vint,interlayer_distance)

	p.plots_cutoff = 3
	update_plots_res(100,p)
	p.plots_n_motifs = 6
	produce_plots = false
	p.plot_for_article = true
	px("N ",N,", Nz ",Nz," d ",p.interlayer_distance)

	# Imports untwisted quantities


	px("W[1,1] ")
	print_low_fourier_modes(p.W_V_plus[1],p;m=3)
	px("W[2,1] ")
	print_low_fourier_modes(p.W_V_plus[3],p;m=3)

	# plot_block_reduced(p.T_BM,p;title="T")
	p.add_non_local_W = true
	# px("Distance between Σ and optimized T_BM ",relative_distance_blocks(p.Σ,p.T_BM)) # MAYBE NOT PRECISE !
	# px("Distance between V_V and optimized T_BM ",relative_distance_blocks(p.𝕍_V,p.T_BM))

	# Compares functions of T and 𝕍
	# px("Comparision to BM")
	# compare_to_BM_infos(p.𝕍_V,p,"V_V") # normal that individual blocks distances are half the total distance because there are two blocks each time
	# compare_to_BM_infos(p.Σ,p,"Σ")

	px("V_{11}(x-(1/3)(a1-a2)) = V_{12}(x) ",distance(translation_interpolation(p.𝕍_V[1], [1/3,-1/3],p),p.𝕍_V[2]))
	px("V_{11}(x+(1/3)(a1-a2)) = V_{12}(x) ",distance(translation_interpolation(p.𝕍_V[1],-[1/3,-1/3],p),p.𝕍_V[2]))

	# plot_block_cart(p.Σ,p;title="Σ")
	# plot_block_cart(p.𝕍_V,p;title="V_V")
	# plot_block_cart(p.T_BM,p;title="T")
	# p.𝕍 = app_block(J_four,p.𝕍,p) # rotates T of J and rescales space of sqrt(3)
	# test_equality_all_blocks(p.Wplus_tot,p;name="W")
	# plot_block_cart(p.W_non_local_plus,p;title="W_nl_plus")

	# testit(p)

	# plot_block_reduced(p.𝕍_V,p;title="V_V")
	# plot_block_reduced(p.Σ,p;title="Σ")
	# plot_block_reduced(T_BM,p;title="T")


	# Mean W
	W = p.compute_Vint ? p.Wplus_tot : p.W_V_plus
	mean_W = mean_block(W,p)
	# means_W_V_minus = mean_block(p.W_V_minus,p)
	# @assert distance(means_W_V_minus,means_W_V_minus)<1e-4
	W_without_mean = add_cst_block(W,-sqrt(p.cell_area)*mean_W,p)
	px("Mean W_V_plus (meV) :")
	display(mean_block(p.W_V_plus,p)*1e3*hartree_to_ev)
	px("\nW_Vint matrix")
	display(p.W_Vint_matrix)
	px("Mean W_plus in meV :")
	display(mean_block(p.Wplus_tot,p)*1e3*hartree_to_ev)

	δ𝕍 = op_two_blocks((x,y)->x.-y,p.𝕍,T_BM_four(p.wAA/p.sqi,p.wAA/p.sqi,p))

	# Plots for article
	if p.plot_for_article
		# plot_block_reduced(p.𝕍_V,p;title="V_V")
		# plot_block_reduced(p.𝕍,p;title="V")
		# plot_block_article(δ𝕍,p;title="δV",k_red_shift=-p.m_q1)
		# plot_block_article(p.T_BM,p;title="T",k_red_shift=-p.m_q1)
		# plot_block_article(W_without_mean,p;title="W_plus_without_mean")
		# plot_block_article(p.𝔸1,p;title="A",other_block=p.𝔸2,k_red_shift=-p.m_q1,meV=false,coef=1/p.vF,vertical_bar=true)
		# if p.compute_Vint plot_block_article(p.𝕍_Vint,p;title="V_Vint",k_red_shift=-p.m_q1) end
		# plot_block_article(p.Σ,p;title="Σ",k_red_shift=-p.m_q1,meV=false)
		# plot_block_article(p.W_non_local_plus,p;title="W_nl_plus",k_red_shift=-p.m_q1,vertical_bar=true)
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

function study_in_d() # curves with and without Vint are extremely close
	N = 27; Nz = 600 # ecut 40|Kd|^2, L = 125
	px("N ",N,", Nz ",Nz)

	p = EffPotentials()
	p.add_non_local_W = true
	import_u1_u2_V_φ(N,Nz,p)
	init_EffPot(p)
	p.compute_Vint = true
	fine = true

	list_d = vcat([0.01],(0.1:0.1:11))
	measures = Dict()
	cf = 4

	meas = ["wAA","wC","wΣ","norm_∇Σ","norm_W_without_mean","distV","distΣ"] #"Wmean"
	id_graph = ["1","1","2","2","2","1","2"]
	wAA_str = "w_{AA}^{d=6.45}"
	ps(s) = string("\\frac{1}{",cf,"} {\\Vert}",s,"\\Vert_{L^2}")
	function cS(a,b=wAA_str;norm=false)
		aa = norm ? ps(a) : a
		LaTeXString(string("\$\\frac{",aa,"}{",b,"}\$")) # s/wAA
	end
	labels = [cS("w_{AA}"),cS("[V_{d} u_1 \\mid u_1]_{d,-1,-1}"),L"$[u_1,u_1]_{d,0,0}$",cS("∇\\Sigma_d","v_F";norm=true),cS("𝕎_d - W_d 𝕀 ";norm=true),cS("𝕍_d - [V u_1,u_1]_{d,0,0} 𝐕 - [V u_1,u_1]_{d,-1,-1} 𝐕";norm=true),LaTeXString(string("\$",ps("\\Sigma_d - [u_1,u_1]_{d,0,0} 𝐕"),"\$"))] # cS("W_d")
	colors = [:blue,:green1,:pink,:red,:black,:brown,:orange,:cyan,:darkgreen]
	for m in meas
		measures[m] = zeros(Float64,length(list_d))
	end

	# print_low_fourier_modes(C_Vu1_u1,p,hartree_to_ev/sqrt(p.cell_area))

	c = 1/sqrt(p.cell_area)

	# Compute wAA for d=6.45
	p.interlayer_distance = 6.45
	import_Vint(p)
	build_blocks_potentials(p)
	build_block_𝔸(p)
	V = p.compute_Vint ? p.𝕍 : p.𝕍_V
	wAA_ref = real(V[1][1,1])

	for i=1:length(list_d)
		p.interlayer_distance = list_d[i]
		import_Vint(p)
		build_blocks_potentials(p)
		build_block_𝔸(p)
		V = p.compute_Vint ? p.𝕍 : p.𝕍_V
		W = p.compute_Vint ? p.Wplus_tot : p.W_V_plus
		wAA = real(V[1][1,1])
		wC = real(V[1][end,2])
		wD = real(V[1][1,2])
		# wE = real(V[1][2,2])
		wΣ = real(p.Σ[1][1,1])

		# W
		mean_W_block = mean_block(W,p)
		W_without_mean = add_cst_block(W,-sqrt(p.cell_area)*mean_W_block,p)
		meanW = real(mean_W_block[1,1])

		measures["wAA"][i] = wAA/wAA_ref
		measures["wC"][i] = wC/wAA_ref
		# measures["wD"][i] = p.sqi*wD
		measures["wΣ"][i] = p.sqi*wΣ
		# measures["wE"][i] = p.sqi*wE

		sm = op_two_blocks((x,y)->x.+y,T_BM_four(wAA,wAA,p),T_BM_four(wC,wC,p;second=true))
		measures["distV"][i] = norm_block(op_two_blocks((x,y)->x.-y,sm,V),p)/(cf*wAA_ref)
		measures["distΣ"][i] = p.sqi*norm_block(op_two_blocks((x,y)->x.-y,T_BM_four(wΣ,wΣ,p),p.Σ),p)/cf
		# measures["norm_V"][i] = p.sqi*norm_block(V)/cf
		# measures["norm_Σ"][i] = p.sqi*norm_block(p.Σ)/cf
		measures["norm_∇Σ"][i] = p.sqi*norm_block_potential(p.𝔸1,p.𝔸2,p)/(cf*p.vF)
		# measures["Wmean"][i] = meanW*sqrt(p.cell_area)/wAA_ref
		measures["norm_W_without_mean"][i] = norm_block(W_without_mean,p)/(cf*wAA_ref)
		px(p.interlayer_distance," ")
	end
	# Print measures
	for i=1:length(meas)
		mea = meas[i]
		px(mea,"\n",measures[mea],"\n")
	end
	function save_fig(fine,graphs_V)
		res = 500
		resX_not_fine = 200
		resX_fine = 350
		meV = fine; c_meV = meV ? 1e3 : 1
		fig = CairoMakie.Figure(resolution=(fine ? resX_fine : resX_not_fine,res))

		ax = CairoMakie.Axis(fig[1, 1], xlabel = "d (Bohr)")#, ylabel = "∅")
		xlim = fine ? maximum(list_d) : 7
		ax.xticks = (0:1:xlim)
		CairoMakie.xlims!(ax,0,xlim)

		maxY2 = 1e2
		minY2 = 1e-5

		if fine
			minY2 = 1e-5
			X = 6.45
			CairoMakie.vlines!(ax,[X],color = :black)
			CairoMakie.annotations!([string("d=",X)], [Point2f0(X+0.2,minY2*1.1)], textsize =20)
			CairoMakie.ylims!(ax,minY2,maxY2)
			ax.yscale = log10
		# else
			# Axis 2
			# CairoMakie.ylims!(ax,-0.1,0.35)
		end
		lin = []
		fun = fine ? abs : x-> x

		for i=1:length(meas)
			mea = meas[i]
			if !fine || (graphs_V && id_graph[i]=="1") || (!graphs_V && id_graph[i]=="2")
				l = CairoMakie.lines!(ax, list_d, fun.(measures[mea]),label=labels[i],color=colors[i],linestyle=nothing)
				push!(lin,l)
			end
		end

		# ax.yaxisposition = :right
		# ax.yticklabelalign = (:left, :center)
		# ax.xticklabelsvisible = false
		# ax.xticklabelsvisible = false
		# ax.xlabelvisible = false
		# CairoMakie.linkxaxes!(ax,ax)
		patchsize = 10

		figlegend = CairoMakie.Figure(resolution=(1000,300))
		if !fine
			CairoMakie.Legend(figlegend[1, 2],lin,labels,framevisible = true,patchsize = (patchsize, patchsize),fontsize=20,nbanks=5)
		end
		add_name = !fine ? "" : (graphs_V ? "_log_V" : "_log_others")
		post_path = string("study_d",add_name,".pdf")
		for pre in ["effective_potentials/",p.article_path]
			CairoMakie.save(string(pre,post_path),fig)
			# CairoMakie.save(string(pre,"legend_study_d.pdf"),figlegend)
		end
	end
	save_fig(true,true)
	save_fig(true,false)
	save_fig(false,true)
end


#################### Third step : compute the bands diagram

# Adimentionalized quantities, wAA and wAB independent
function explore_band_structure_BM_TKV()
	p = Basis()
	p.N = 8
	# @assert mod(p.N,2)==1 # for more symmetries
	p.a = 4π/sqrt(3)
	p.dim = 2
	p.l = 15 # number of eigenvalues we compute
	init_basis(p)
	p.resolution_bands = 5
	p.energy_unit_plots = "Hartree"
	p.folder_plots_bands = "bands_BM"
	p.energy_center = 0
	p.energy_scale = 1.5
	p.solver = "Exact"
	p.coef_derivations = 1
	p.plots_article = false

	for q in [p.q1,p.q2,p.q3] q ./= norm(q) end
	update_a(p.q2-p.q1,p.q3-p.q2,p)

	K1 = [1,2]/3; K2 = [-1,1]/3
	Γ = [0.0,0.0]; Γ2 = 2*K1-K2; M = Γ2/2

	Klist = [K2,K1,Γ2,M,Γ]
	Klist_names = ["K2","K1","Γ'","M","Γ"]
	# plot_path(Klist,Klist_names,p)
	# Klist = [K2,Γ,M]; Klist_names = ["K2","Γ","M"]
	valleys = [1;-1]
	Kf = [k -> Dirac_k(k,p;valley=valleys[i],K1=K1,K2=K2) for i=1:2]

	output = "BM"
	αEgβ = true # α = β or α = 0

	θs = [1.05]
	βs = [0.605]

	w = 110*1e-3*ev_to_hartree
	if output=="BM"
		θrads = (π/180)*θs
		kθs = 2*sin.(θrads/2)*4π/(4.66*3)
		βs = w./(kθs.*p.vF)
		p.energy_scale = 250
	else
		# βs = [0.586]
		# βs = vcat([0.586],(0:0.05:0.8))
	end
	for i in length(βs)
		if output=="BM" p.coef_energies_plot = hartree_to_ev*1e3*kθs[i]*p.vF end # energies in meV

		β = βs[i]
		α = αEgβ ? β : 0
		print("(α,β)=(",α,",",β,")")
		T0 = T_BM_four(α,β,p)
		T = a2c(T0,p)
		TBMs = [T,app_block((M,p)->conj.(M),T,p)]
		Ts = [TBMs[1],app_block(parity_four,TBMs[2],p)]
		Ts_full = [build_offdiag_V(Ts[i],p) for i=1:2]#*sqrt(p.cell_area)
		σs = [spectrum_on_a_path(Ts_full[i],Kf[i],Klist,p) for i=1]

		θ = θs[i]
		title = output=="BM" ? θ : β
		pl = plot_band_diagram([σs[1]],Klist,Klist_names,p;post_name="bm",title=title)
	end
	p
end

function explore_band_structure_Heff()
	# N = 24 : does it
	# N = 12; Nz = 75
	# N = 15; Nz = 270
	# N = 24; Nz = 180
	# N = 20; Nz = 150
	# N = 27; Nz = 576 # ecut
	N = 27; Nz = 600 # <---
	interlayer_distance = 6.45
	# do_BM = false
	job = "bandwidths"
	# job = "diagram"

	# Imports u1, u2, V, Vint, v_fermi and computes the effective potentials
	compute_Vint = true
	EffV = import_and_computes(N,Nz,compute_Vint,interlayer_distance)

	reduce_N(EffV,9) # Because the initial dimensions are too high to compute

	p = Basis()
	p.N = EffV.N
	# @assert mod(p.N,2)==1 # for more symmetries
	p.a = 4.66
	p.dim = 2
	p.l = 15 # number of eigenvalues we compute
	init_basis(p)

	p.resolution_bands = 5
	if job=="bandwidths" p.resolution_bands = 10 end
	p.folder_plots_bands = "eff"
	p.energy_scale = 1.5
	p.solver = "Exact"
	p.coef_derivations = 1
	p.plots_article = true

	# update_a(p.q2-p.q1,p.q3-p.q2,p) # norm(qj) = kD = 4π/(3a)

	# K1 = [1,2]/3; K2 = [-1,1]/3
	K1 = [-1,2]/3; K2 = [-2,1]/3
	Γ = [0.0,0.0]; Γ2 = 2*K1-K2; M = Γ2/2

	Klist = [K2,K1,Γ2,M,Γ]; Klist_names = ["K_2","K_1","Γ'","M","Γ"]
	if job=="bandwidths" Klist = [K1,M,Γ]; Klist_names = ["K_1","M","Γ"] end
	# if job!="bandwidths" plot_path(Klist,Klist_names,p) end
	# Klist = [K2,Γ,M]; Klist_names = ["K2","Γ","M"]
	valleys = [1;-1]

	# k-dependent part
	(A1,A2) = build_mag_block(EffV;Q=-EffV.q1_red,J=false)
	(JA1,JA2) = build_mag_block(EffV;Q=-EffV.q1_red,J=true)

	multiply_potentials(p.sqi,EffV)

	# Build T
	EffV.wAA = 0.11*ev_to_hartree
	T = T_BM_four(EffV.wAA,EffV.wAA,EffV)
	print_wAA(EffV)
	# T = p.sqi*EffV.T_BM
	Tm = build_offdiag_V(T,p)

	# Build SΣ
	SΣ = build_offdiag_V(EffV.Σ,p)
	S = Hermitian(I + 1*SΣ)
	p.ISΣ = Hermitian(inv(sqrt(S)))
	# p.ISΣ = I
	
	# Shifts the Fermi energy for plots
	coef_W = 1
	W = EffV.compute_Vint ? EffV.Wplus_tot : EffV.W_V_plus
	mW = coef_W*real(mean_block(W,EffV)[1,1])
	p.energy_center = 0
	# p.energy_center = mW*hartree_to_ev*1e3
	p.energy_scale = 250

	# Kinetic operators
	Kf_pure(K) = p.vF*Dirac_k(K,p;K1=K1,K2=K2)
	Kf_ours(K,c,ε) = p.ISΣ*(
				c*p.vF*Dirac_k(K,p;K1=K1,K2=K2)
				+ c*offdiag_A_k(EffV.J𝔸1,EffV.J𝔸2,K,p;K1=K1,K2=K2)
				+ (1/2)*ε*(p.vF*Dirac_k(K,p;K1=K1,K2=K2,coef_1=-1,J=true)
					   + offdiag_A_k(EffV.𝔸1,EffV.𝔸2,K,p;K1=K1,K2=K2)
					   + ondiag_mΔ_k(K,p;K1=K1,K2=K2)
					   + offdiag_mΔ_k(EffV.Σ,K,p;K1=K1,K2=K2)
					   )
				# VVERIFIER QUE TOUT EST BIEN HERMITIEN !!!!!!!!!!!!!!
				)*p.ISΣ

	# Constant operator
	cst_op_ours = p.ISΣ*( build_offdiag_V(EffV.𝕍,p) + coef_W*build_ondiag_W(EffV.Wplus_tot,EffV.Wminus_tot,p) )*p.ISΣ

	σs = []
	function bands(ours_or_bm,θ)
		θrad = (π/180)*θ
		εθ = 2*sin.(θrad/2)
		cθ = cos(θrad/2)
		p.coef_energies_plot = hartree_to_ev*1e3*εθ

		px("Computes band diagram, (N,d,θ)=(",p.N,",",interlayer_distance,",",θ,")")
		ours = ours_or_bm=="ours"
		V = (1/εθ)*(ours ? cst_op_ours : Tm)

		kin = ours ? k -> Kf_ours(k,cθ,εθ) : Kf_pure
		σ = spectrum_on_a_path(V,kin,Klist,p;print_progress=true)

		σ
	end

	θs_magic_bm = [1.3425]
	# θs_magic_bm = [1.3425,0.561]#,0.337] # wAA = 126 meV
	θs_magic_ours = [1.150,0.463] # ,0.445]
	θs = vcat((0.4:0.1:1.5),θs_magic_bm,θs_magic_ours)
	θs = vcat((1.15:0.02:1.25))
	# θs = [θs_magic_bm[1],θs_magic_ours[1]]
	# θs = vcat((0.556:0.0005:0.565))
	sort!(θs)

	function compute_bandwidths(θs)
		bw_bm = zeros(length(θs)); bw_ours = zeros(length(θs))
		for i=1:length(θs)
			θ = θs[i]
			# σ_ours = bands("ours",θ)
			σ_bm = bands("bm",θ)
			σ_ours = σ_bm
			# σ_bm = σ_ours

			bw_bm[i] = bandwidth(σ_bm,p)
			bw_ours[i] = bandwidth(σ_ours,p)
			px("θ ",θ," bandwidths ",bw_bm[i]*coef_plot_meV(θ,p)," ",bw_ours[i]*coef_plot_meV(θ,p)," meV")
		end
		plot_bandwidths(θs,bw_bm,bw_ours,p)
	end
	if job=="bandwidths"
		compute_bandwidths(θs)
	elseif job=="diagram"
		# for θ in [θs_magic_bm[2]]#,θs_magic_ours[2]]
			σ_ours = bands("ours",θs_magic_ours[1])
			σ_bm = bands("bm",θs_magic_bm[1])
			σs = [σ_bm,σ_ours]
			θs = [θs_magic_bm[1],θs_magic_ours[1]]
			nmid = fermi_label(p)
			moy = (σ_ours[1,nmid] + σ_ours[1,nmid+1])/2
			shifts = [0,-moy]
			plot_band_diagram([σs[1]],[θs[1]],Klist,Klist_names,"",p;post_name="bm",shifts=[shifts[1]],colors=[:black])
			plot_band_diagram([σs[2]],[θs[2]],Klist,Klist_names,"",p;post_name="eff",shifts=[shifts[2]],colors=[:red])

			bw_bm = bandwidth(σ_bm,p)*coef_plot_meV(θs_magic_bm[1],p)
			bw_ours = bandwidth(σ_ours,p)*coef_plot_meV(θs_magic_ours[1],p)
			px(" bandwidths BM: ",bw_bm," OURS: ",bw_ours)
		# end
	end

	p
end

# p = computes_and_plots_effective_potentials()
explore_band_structure_Heff()
# explore_band_structure_BM_TKV()
# study_in_d()
nothing

#### Todo
# Donner les valeurs des coefs et pas des valeurs absolues pour voir à quoi c'est égal
# Donner les normes L2 des matrices S, S^1 etc
# Ecrire aevc L2(u1 u1 S^0) = |u1 u1| 4.?
# VERIFIER LE PB AVEC VINT DECALE !

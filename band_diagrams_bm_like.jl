include("misc/lobpcg.jl")
include("common_functions.jl")
include("misc/create_bm_pot.jl")
# include("misc/plot.jl")
include("effective_potentials.jl")
using LinearAlgebra, JLD, FFTW, Plots

mutable struct Basis
	# Parameters of the 2d cell
	a; a1; a2; a1_star; a2_star
	x_axis_cart
	dx; dS; dv
	k_axis; k_grid
	K_red
	N
	cell_area

	# Useless parameters
	Nz; L; dz
	kz_axis
	rotate_cell

	# Parameters for the 4×4 matrix
	N2d; Mfull
	Li_tot; Li_k; Ci_tot; Ci_k; k2lin
	S # part_hole matrix
	l; l1; l2; n_l
	ISΣ # S^{-1/2}
	solver
	Iα; Iβ
	fermi_velocity
	double_dirac # true if 4×4 matrix, false if 2×2 matrix
	coef_derivations # applies a coefficient to all derivation operator

	# Parameters for plots
	root_path
	energy_center; energy_scale
	folder_plots_bands
	folder_plots_matrices
	resolution_bands
	energy_unit_plots # ∈ ["eV","Hartree"]
	hartree_to_ev
	ev_to_hartree
	function Basis()
		p = new()
		p.double_dirac = true
		p
	end
end

energy_factor(p) = p.energy_unit_plots == "eV" ? p.hartree_to_ev : 1

function init_basis(p)
	### Parameters of the 2d cell
	p.k_axis = Int.(fftfreq(p.N)*p.N)
	p.root_path = "band_diagrams_bm_like/"
	create_dir(p.root_path)

	p.folder_plots_matrices = "matrices/"
	path2 = string(p.root_path,p.folder_plots_matrices)
	create_dir(path2)
	p.rotate_cell = false
	init_cell_vectors(p;rotate=p.rotate_cell)
	p.S = nothing

	### Parameters for the 4×4 matrix
	number_of_states = p.double_dirac ? 4 : 2
	p.N2d = p.N^2; p.Mfull = number_of_states*p.N2d

	ref = zeros(number_of_states,p.N,p.N); ref_k = zeros(p.N,p.N)
	p.Li_tot = LinearIndices(ref);    p.Li_k = LinearIndices(ref_k)
	p.Ci_tot = CartesianIndices(ref); p.Ci_k = CartesianIndices(ref_k)
	init_klist2(p)

	p.ISΣ = -1 # matrix (1+S_Σ)^(-1/2)
	p.solver = "LOBPCG"
	p.fermi_velocity = 0.38
	p.hartree_to_ev = 27.2114
	p.ev_to_hartree = 1/p.hartree_to_ev

	# Matrices to apply weights
	init_matrices_weights(p)

	# Eigenvalues
	p.l1 = floor(Int,p.Mfull/2) - p.l
	p.l2 = floor(Int,p.Mfull/2) + p.l
	p.n_l = 2*p.l + 1
end

reload_a(p) = init_cell_vectors(p;rotate=p.rotate_cell)

######################### Main functions, to plot band diagrams




function plot_band_structure(Hv,Kdep,name,p)
	Γ = [0,0.0]
	K = p.K_red
	M = [0,1/2]

	A = [1/3,2/3] # K'
	B = p.K_red # K
	C = 2 .*B .- A # Γ1
	M = C/2
	D = Γ
	Klist = [A,B,C,M,D]; Klist_names = ["A","B","C","M","D"]
	# Klist = [K,Γ,M]; Klist_names = ["K","Γ","M"]
	Klist = [Γ,K,M]; Klist_names = ["Γ","K","M"]
	# Klist = [Γ,M,K]; Klist_names = ["M","K","Γ"]


	# Klist = [Klist[i] .- p.K_red for i=1:length(Klist)] # don't do that
	σ_on_path = spectrum_on_a_path(Hv,Kdep,Klist,p)
	pl = plot_band_diagram(σ_on_path,Klist,Klist_names,p)#;K_relative=p.K_red)
	path = string(p.root_path,p.folder_plots_bands)
	create_dir(path)

	s = string(name,"000000000")
	title = s[1:min(6,length(s))]
	savefig(pl,string(path,"/band_struct_",title,".png"))
end

function do_one_value(HvK,p)
	(E,Xf) = solve_one(HvK,p)
	pl = plot_spectrum(E)
	savefig(pl,"spectrum_K.png")
end


######################### Coordinates change

function lin2coord(c_lin,p)
	ci = p.Ci_tot[c_lin]
	(ci[1],ci[2],ci[3]) # (i,pix,piy)
end

function lin_k2coord_ik(c_lin,p)
	ci = p.Ci_k[c_lin]
	(ci[1],ci[2]) # (pix,piy)
end

k_axis(pix,piy,p) = (p.k_axis[pix],p.k_axis[piy])
vec2C(k) = k[1]+im*k[2]

# pix,piy is a 2d moment (the labels) in reduced coords
function coords_ik2full_i_lllll(mi1,mi2,p)
	if p.double_dirac
		return (p.Li_tot[1,mi1,mi2],p.Li_tot[2,mi1,mi2],p.Li_tot[3,mi1,mi2],p.Li_tot[4,mi1,mi2])
	else
		return (p.Li_tot[1,mi1,mi2],p.Li_tot[2,mi1,mi2])
	end
end

function coords_ik2full_i(mi1,mi2,p)
	(p.Li_tot[1,mi1,mi2],p.Li_tot[2,mi1,mi2],p.Li_tot[3,mi1,mi2],p.Li_tot[4,mi1,mi2])
end



######################### Functions coordinates change

# from the solution of LOBPCG ψ_lin[4*p.N^2] to 4 vectors containing the fourier transforms
function lin2four(ψ_lin,p) 
	ψ_four = init_vec(p)
	for lin=1:p.Mfull
		(i,pix,piy) = lin2coord(lin,p)
		ψ_four[i][pix,piy] = ψ_lin[lin]
	end
	ψ_four
end

function init_klist2(p)
	p.k2lin = []
	for c_lin=1:p.Mfull
		(i,pix,piy) = lin2coord(c_lin,p)
		push!(p.k2lin,p.k_axis[pix]^2 + p.k_axis[piy]^2)
	end
end

######################### Derivation operators

# Gives the action (hat{VX})_k in Fourier space. Vfour is the Fourier transform of V, Xfour is the Fourier transform of X
# actionV(Vfour,Xfour,p) = vcat(cyclic_conv(Vfour,Xfour)/length(Vfour)...)
# actionH(Vfour,p) = X -> p.k2lin.*X .+ actionV(Vfour,Reshape(X,p),p) # X = Xlin, v in direct space

# Creates [σ(-i∇+k)        0]
#         [0        σ(-i∇+k)]
function Dirac_k(κ,p) # κ in reduced coordinates, κ_cart = κ_red_1 a1_star + κ_red_2 a2_star
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	kC = vec2C(κ[1]*p.a1_star + κ[2]*p.a2_star)
	for ck_lin=1:p.N2d
		(mi1,mi2) = lin_k2coord_ik(ck_lin,p)
		(m1,m2) = k_axis(mi1,mi2,p)
		vC = vec2C(m1*p.a1_star + m2*p.a2_star)*p.coef_derivations + kC
		(c1,c2,c3,c4) = coords_ik2full_i(mi1,mi2,p)
		H[c1,c2] = conj(vC); H[c2,c1] = vC; H[c3,c4] = conj(vC); H[c4,c3] = vC
	end
	# test_hermitianity(H,"Kinetic Dirac part")
	Hermitian(H)
end


# Creates (-i∇+k)^2 𝕀_{4×4}
function mΔ(k,p)
	n = p.Mfull
	Δ = zeros(ComplexF64,n,n)
	kC = k[1]*p.a1_star + k[2]*p.a2_star
	for ck_lin=1:p.N2d
		(mi1,mi2) = lin_k2coord_ik(ck_lin,p)
		(c1,c2,c3,c4) = coords_ik2full_i(mi1,mi2,p)
		(m1,m2) = k_axis(mi1,mi2,p)
		vC = norm((m1*p.a1_star + m2*p.a2_star)*p.coef_derivations .+ kC)^2
		Δ[c1,c1],Δ[c2,c2],Δ[c3,c3],Δ[c4,c4] = vC,vC,vC,vC
	end
	# test_hermitianity(Δ,"Δ order 1")
	# save_H(Δ,"Δ",p)
	Δ
end

# Creates [-σ⋅J(-i∇+k)           0]
#         [0            σ⋅J(-i∇+k)]
function J_Dirac_k(k,p)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	kC = vec2C(k[1]*p.a1_star + k[2]*p.a2_star)
	for ck_lin=1:p.N2d
		(mi1,mi2) = lin_k2coord_ik(ck_lin,p)
		(c1,c2,c3,c4) = coords_ik2full_i(mi1,mi2,p)
		(m1,m2) = k_axis(mi1,mi2,p)
		vC = vec2C(m1*p.a1_star + m2*p.a2_star)*p.coef_derivations + kC
		H[c1,c2] = im*conj(vC)
		H[c2,c1] = -im*vC
		H[c3,c4] = -im*conj(vC)
		H[c4,c3] = im*vC
	end
	# test_hermitianity(H,"Dirac order 1")
	# save_H(H,"H",p)
	H
end

######################### Electric potential operators

# Creates [0  𝕍]
#         [𝕍* 0]
function V_offdiag_matrix(v0,p) # v = [v1,v2,v3,v4] = mat(v1 & v2 \\ v3 & v4), Fourier coeffcients
	v = lin2mat(v0)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	for n_lin=1:n
		(α,ni1,ni2) = lin2coord(n_lin,p)
		(n1,n2) = k_axis(ni1,ni2,p)
		for m_lin=1:n
			(β,mi1,mi2) = lin2coord(m_lin,p)
			(m1,m2) = k_axis(mi1,mi2,p)
			Pi1,Pi2 = k_inv(n1-m1,n2-m2,p)
			Ki1,Ki2 = k_inv(m1-n1,m2-n2,p)
			c = 0
			if α ≥ 3 && β ≤ 2
				c = conj(v[β,α-2][Ki1,Ki2])
			elseif α ≤ 2 && β ≥ 3
				c = v[α,β-2][Pi1,Pi2]
			end
			H[n_lin,m_lin] = c
		end
	end
	# test_hermitianity(H,"offdiag V matrix")
	# test_part_hole_sym_matrix(H,p,"H")
	# save_H(H,"potential_V",p)
	# display([H[mod1(x,p.Mfull),mod1(y,p.Mfull)] for x=1:30, y=1:30])
	Hermitian(H)/sqrt(p.cell_area)
end

# Creates [Vp  0 ]
#         [0   Vm]
function V_ondiag_matrix(Vp0,Vm0,p)
	Vp = lin2mat(Vp0); Vm = lin2mat(Vm0)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	for n_lin=1:n
		(α,ni1,ni2) = lin2coord(n_lin,p)
		(n1,n2) = k_axis(ni1,ni2,p)
		for m_lin=1:n
			(β,mi1,mi2) = lin2coord(m_lin,p)
			(m1,m2) = k_axis(mi1,mi2,p)

			c = 0
			Pi1,Pi2 = k_inv(n1-m1,n2-m2,p)
			if α ≤ 2 && β ≤ 2
				c = Vp[α,β][Pi1,Pi2]
			elseif α ≥ 3 && β ≥ 3
				c = Vm[α-2,β-2][Pi1,Pi2]
			end
			H[n_lin,m_lin] = c
		end
	end
	# test_hermitianity(H,"ondiag V matrix")
	# save_H(H,"potential_ondiag_V",p)
	# test_part_hole_sym_matrix(Hk,p,"Hvk")
	# display([H[mod1(x,p.Mfull),mod1(y,p.Mfull)] for x=1:30, y=1:30])
	Hermitian(H)
end

######################### Magnetic potential/derivation operators

# Creates [0          𝔸⋅(-i∇+k)]
#         [𝔸*⋅(-i∇+k)         0]
function A_offdiag_matrix(Aa1,Aa2,k,p)
	A1 = lin2mat(Aa1); A2 = lin2mat(Aa2)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	K_cart = k[1]*p.a1_star + k[2]*p.a2_star
	k1 = K_cart[1]; k2 = K_cart[2]
	for n_lin=1:n
		(α,ni1,ni2) = lin2coord(n_lin,p)
		(n1,n2) = k_axis(ni1,ni2,p)
		for m_lin=1:n
			(β,mi1,mi2) = lin2coord(m_lin,p)
			(m1,m2) = k_axis(mi1,mi2,p)

			c = 0
			Pi1,Pi2 = k_inv(n1-m2,  n2-m2,p)
			Ki1,Ki2 = k_inv(-n1+m1,-n2+m2,p)

			m1 *= p.coef_derivations
			m2 *= p.coef_derivations

			if α ≥ 3 && β ≤ 2
				c = (m1*p.a1_star[1]+m2*p.a2_star[1]+k1)*conj(A1[β,α-2][Ki1,Ki2])
				+ (m1*p.a1_star[2]+m2*p.a2_star[2]+k2)*conj(A2[β,α-2][Ki1,Ki2])
			elseif α ≤ 2 && β ≥ 3
				c = (m1*p.a1_star[1]+m2*p.a2_star[1]+k1)*A1[α,β-2][Pi1,Pi2]
				+ (m1*p.a1_star[2]+m2*p.a2_star[2]+k2)*A2[α,β-2][Pi1,Pi2]
			end
			H[n_lin,m_lin] = c
		end
	end
	# test_hermitianity(H,"A∇")
	# test_part_hole_sym_matrix(H,p,"A∇")
	# save_H(H,"potential_V",p)
	# display([H[mod1(x,p.Mfull),mod1(y,p.Mfull)] for x=1:30, y=1:30])
	Hermitian(H)
end

# Creates [0            V(-i∇+k)^2]
#         [V^*(-i∇+k)^2          0]
function VΔ_offdiag_matrix(V0,k,p)
	V = lin2mat(V0)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	K_cart = k[1]*p.a1_star + k[2]*p.a2_star
	for n_lin=1:n
		(α,ni1,ni2) = lin2coord(n_lin,p)
		(n1,n2) = k_axis(ni1,ni2,p)
		for m_lin=1:n
			(β,mi1,mi2) = lin2coord(m_lin,p)
			(m1,m2) = k_axis(mi1,mi2,p)
			c = 0
			Pi1,Pi2 = k_inv(n1-m2,  n2-m2,p)
			Ki1,Ki2 = k_inv(-n1+m1,-n2+m2,p)
			if α ≥ 3 && β ≤ 2
				c = conj(V[β,α-2][Ki1,Ki2])
			elseif α ≤ 2 && β ≥ 3
				c = V[α,β-2][Pi1,Pi2]
			end
			r = norm((m1*p.a1_star+m2*p.a2_star)*p.coef_derivations + K_cart)^2
			H[n_lin,m_lin] = r*c
		end
	end
	# test_hermitianity(H,"ΣΔ")
	# test_part_hole_sym_matrix(H,p,"ΣΔ")
	# save_H(H,"potential_V",p)
	# display([H[mod1(x,p.Mfull),mod1(y,p.Mfull)] for x=1:30, y=1:30])
	Hermitian(H)
end

# Creates σ(-i∇+k)
function free_Dirac_k_monolayer(κ,p)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	kC = vec2C(κ[1]*p.a1_star + κ[2]*p.a2_star)
	for ck_lin=1:p.N2d
		(mi1,mi2) = lin_k2coord_ik(ck_lin,p)
		(m1,m2) = k_axis(mi1,mi2,p)
		vC = vec2C(m1*p.a1_star + m2*p.a2_star)*p.coef_derivations + kC
		(c1,c2) = coords_ik2full_i(mi1,mi2,p)
		H[c1,c2] = conj(vC); H[c2,c1] = vC
	end
	# display(H)
	# test_hermitianity(H,"Kinetic Dirac part")
	# save_H(H,p,"free_dirac")
	# rhm = heatmap(real.(H))
	# ihm = heatmap(imag.(H))
	# ahm = heatmap(abs.(H))
	# pl = plot(rhm,ihm,ahm,size=(1000,700))
	# savefig(pl,"free_dirac.png")
	Hermitian(H)
end

# Creates (-i∇+k)^2 𝕀_2×2
function free_Schro_k_monolayer(κ,p)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	kC = κ[1]*p.a1_star + κ[2]*p.a2_star
	for ck_lin=1:p.N2d
		(mi1,mi2) = lin_k2coord_ik(ck_lin,p)
		(m1,m2) = k_axis(mi1,mi2,p)
		v = norm((m1*p.a1_star + m2*p.a2_star)*p.coef_derivations + kC)^2
		(c1,c2) = coords_ik2full_i(mi1,mi2,p)
		H[c1,c1] = v; H[c2,c2] = v
	end
	display(H)
	# test_hermitianity(H,"Kinetic Dirac part")
	# save_H(H,p,"free_dirac")
	# rhm = heatmap(real.(H))
	# ihm = heatmap(imag.(H))
	# ahm = heatmap(abs.(H))
	# pl = plot(rhm,ihm,ahm,size=(1000,700))
	# savefig(pl,"free_dirac.png")
	Hermitian(H)
end

######################### Add weights

# adds weights V = [a b ; c d] -> [αa βb ; βc αd]
function weights_off_diag_matrix(V,α,β,p)
	@assert abs(imag(α)) + abs(imag(β)) < 1e-10
	(α-1)*p.Iα.*V .+ (β-1)*p.Iβ.*V .+ V
end

function init_matrices_weights(p) # adds weights V = [a b ; c d] -> [αa βb ; βc αd]
	n = p.Mfull
	p.Iα = zeros(ComplexF64,n,n); p.Iβ = zeros(ComplexF64,n,n)
	for n_lin=1:n
		(I,ni1,ni2) = lin2coord(n_lin,p)
		for m_lin=1:n
			(J,mi1,mi2) = lin2coord(m_lin,p)
			if (I,J) in [(1,3),(2,4),(3,1),(4,2)]
				p.Iα[n_lin,m_lin] = 1
			elseif (I,J) in [(1,4),(2,3),(3,2),(4,1)]
				p.Iβ[n_lin,m_lin] = 1
			end
		end
	end
	# test_hermitianity(H,"V matrix")
	# save_H(H,"potential_V",p)
	# display([H[mod1(x,p.Mfull),mod1(y,p.Mfull)] for x=1:30, y=1:30])
end

######################### Finds lower eigenmodes

# Applies LOBPCG
function apply_lobpcg(H,l,p,X0,maxiter=100,tol=1e-6)
	L = X -> H*X
	(λs,φs,cv,Xf) = solve_lobpcg(L,l,p.k2lin;maxiter=maxiter,tol=tol,X0=X0,full_diag=true)
	ψs_four = [lin2four(φs[i],p) for i=1:l]
	(λs,ψs_four,cv,Xf)
end

# Solve H_K for one K
function solve_one(HvK,p,X0=-1) # l is the number of eigenvalues we want
	X = []
	if X0 == -1
		X = randn(ComplexF64,size(HvK,1),p.l2)
	else
		X = X0
	end
	if p.solver=="LOBPCG"
		(E,φs,c,Xf) = apply_lobpcg(HvK,p.l2,p,X)
	else
		(E,φs) = eigen(HvK)
		E = E[1:p.l2]; φs = φs[:,1:p.l2] # Peut etre pas bon pour φs mais on s'en sert pas
		# px(E)
		Xf = -1
	end
	(E,Xf)
end


######################### Computes the band diagram

# Computes spectrum for eigenvalues between l1 and l2
# It is paralellized
function spectrum_on_a_path(Hv,Kdep,Klist,p)
	res = p.resolution_bands
	n = length(Klist)
	T = [i/res for i=0:res-1]
	n_path_points = res*n
	n_eigenvals = 1
	graphs = zeros(n_path_points,n_eigenvals)
	X = -1
	graphs = zeros(n_path_points,p.n_l)
	for Ki=1:n
		K0 = Klist[Ki]; K1 = Klist[mod1(Ki+1,n)]
		path = [(1-t/res)*K0 .+ (t/res)*K1 for t=0:res-1]
		# Threads.@threads for s=1:res
		for s=1:res
			HvK = Hv + Kdep(path[s])
			# px("K ",K0," ",K1)
			# test_hermitianity(HvK,"Hvk")
			# test_part_hole_sym_matrix(HvK,p,"Hvk")
			(E,Xf) = solve_one(Hermitian(HvK),p,X)
			# Selects eigenvalues around the Fermi level
			E = E[p.l1:p.l2]
			X = Xf
			indice = (Ki-1)*res + s
			graphs[indice,:] = E
			percentage = 100*(Ki*(res-1)+s-1)/((n)*(res))
			# px("Percentage done ",percentage)
		end
	end
	graphs
end

# From the numbers of the band diagram, produces a plot of it
function plot_band_diagram(graphs,Klist,Knames,p;K_relative=[0.0,0.0])
	n = length(Klist)
	res = p.resolution_bands
	n_path_points = res*n
	ylims = p.energy_center-p.energy_scale,p.energy_center+p.energy_scale
	lengths_paths = [norm(k_red2cart(Klist[mod1(i+1,n)]-K_relative,p) .- k_red2cart(Klist[i]-K_relative,p)) for i=1:n]
	lengths_paths /= sum(lengths_paths)
	dx_list = lengths_paths/res
	x_list = []; starts_x = []; end_x = 0
	for j=1:n
		dx = dx_list[j]
		cur_l = end_x .+ [dx*(i-1) for i=1:res]
		x_list = vcat(x_list,cur_l)
		push!(starts_x,end_x)
		end_x += res*dx
	end
	pl = Plots.plot(size=(1000,1100),ylims=ylims,legend=false) #:topright)
	for l=1:p.n_l
		Plots.plot!(pl,x_list,graphs[:,l]*energy_factor(p),xticks=nothing)
	end
	colors = [:green,:cyan,:blue,:red,:yellow]
	if length(colors) < length(Knames)
		px("NOT ENOUGH COLORS IN plot_band_diagram")
	end
	for Ki=1:n
		x = starts_x[Ki]
		Plots.plot!(pl,[x], seriestype="vline", label=Knames[Ki], color=colors[Ki])
		annotate!(x+0.01, ylims[1]+(ylims[2]-ylims[1])/20, Plots.text(Knames[Ki], colors[Ki], :left, 20))
	end
	pl
end

######################### Symmetry tests

# Matrix of the particle-hole symmetry
function part_hole_matrix(p)
	n = p.Mfull
	S = zeros(ComplexF64,n,n)
	for n_lin=1:n
		(α,ni1,ni2) = lin2coord(n_lin,p)
		(n1,n2) = k_axis(ni1,ni2,p)
		for m_lin=1:n
			(β,mi1,mi2) = lin2coord(m_lin,p)
			(m1,m2) = k_axis(mi1,mi2,p)

			Pi1,Pi2 = k_inv(n1+m1,n2+m2,p)
			c = 0
			if Pi1 == 1 && Pi2 == 1
				if α ≥ 3 && β ≤ 2 && α == β+2
					c = 1
				elseif α ≤ 2 && β ≥ 3 && α == β-2
					c = -1
				end
			end
			S[n_lin,m_lin] = im*c
		end
	end
	# test_hermitianity(S,"part_hole matrix")
	# save_H(S,"part_hole_mat",p)
	# display([H[mod1(x,p.Mfull),mod1(y,p.Mfull)] for x=1:30, y=1:30])
	Hermitian(S)
end

function test_part_hole_sym_matrix(L,p,name="")
	if p.S == nothing
		p.S = part_hole_matrix(p)
	end
	s = sum(abs.(L))
	c = s < 1e-10 ? 0 : sum(abs.(p.S*L*p.S + L))/s
	px(string("Test part_hole sym at matrix level ",name,": "),c)
end

######################### Plots

function save_H(H,p,title="H")
	rhm = heatmap(real.(H))
	ihm = heatmap(imag.(H))
	ahm = heatmap(abs.(H))
	pl = plot(rhm,ihm,ahm,size=(1000,700))
	savefig(pl,string(p.root_path,title,".png"))
end

function plots_BM_pot(α,β,p)
	Vbm_four = Veff_BM(p.N,p.a,α,β,true)
	Vbm_four = ifft.(Veff_BM(p.N,p.a,α,β,true))
	Vbm_direct = Veff_BM(p.N,p.a,α,β,false)
	# Vbm_direct = fft.(Veff_BM(p.N,p.a,α,β,false))
	plotVbm(Vbm_four,Vbm_direct,p)
end

########################## Plot Hamiltonians an other matrices

function swap_table(M) # to have the matrix representation where [1,1] is on the top left
	n = size(M,2); m = size(M,1)
	[M[m-j+1,i] for i=1:n, j=1:m]
end

function verification_heatmap(p)
	M = [1 2;3 4]
	plot_heatmap(M,"test_heatmap")
end

function plot_heatmap(M,name,p)
	if sum(abs.(M))<1e-10
		px("Makie heatmap is not able to plot colorbars of 0 matrices")
		return 0
	end
	fig = Figure()
	f, ax = Makie.heatmap(fig[1,1],swap_table(M))
	Colorbar(fig[1,2], ax)
	path = string(p.root_path,p.folder_plots_matrices)
	save(string(path,name,".png"),fig)
end

######################### Low level functions

init_vec(p) = fill2d(zeros(ComplexF64,p.N,p.N),4)


######################### Archive of another implementation, not used

# Creates [σ(-i∇+k)        0]
#         [0        σ(-i∇+k)]
function Dirac_k2(k,p) # κ in reduced coordinates, κ_cart = κ_red_1 a1_star + κ_red_2 a2_star
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	for mi1=1:p.N, mi2=1:p.N
		v = [mi1,mi2]
		D = build_block_Dirac(v,k,p)
		fill_block_m(v,v,H,D,p)
	end
	# test_hermitianity(H,"Dirac_k")
	Hermitian(H)
end

# A is a Mfull×Mfull matrix, B is a 4×4 matrix, m and n are in [1,p.N]^2 and correspond to the Fourier labels
function fill_block_m(m,n,A,B,p)
	I = im2ilin(m,p)
	J = im2ilin(n,p)
	for i=0:3, j=0:3
		A[I+i,J+j] = B[i+1,j+1]
	end
end

# builds a block 4×4 with
#  [σ(-i∇+ka*)        0]
#  [0        σ(-i∇+ka*)]
function build_block_Dirac(mi,k,p)
	(m1,m2) = k_axis(mi[1],mi[2],p)
	c = vec2C((m1+k[1])*p.a1_star + (m2+k[2])*p.a2_star)
	cc = conj(c)
	[0 cc 0 0 ;
	 c 0  0 0 ;
	 0 0  0 cc;
	 0 0  c 0  ]
end

function k_inv_new_1d(m,p)
	if -1 ≤ m ≤ 1
		return mod(m,p.N)+1
	end
	return nothing
end


# Creates [0  𝕍]
#         [𝕍* 0]
function V_at_mi(i,j,V)
	[0                   0                    V[1][i,j] V[2][i,j];
	 0                   0                    V[3][i,j] V[4][i,j];
	 conj(V[1][i,j]) conj(V[3][i,j])  0             0            ;
	 conj(V[2][i,j]) conj(V[4][i,j])  0             0             ]
end

function swap2(M) # to have the matrix representation where [1,1] is on the top left
	n = size(M,2); m = size(M,1)
	[M[m-i+1,j] for i=1:m, j=1:n]
end

function V_offdiag_matrix2(V,p)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	for mi1=1:p.N, mi2=1:p.N, ni1=1:p.N, ni2=1:p.N
		Mi = [mi1,mi2]; Ni = [ni1,ni2]

		# (ℓi1,ℓi2) = k_inv(mi[1]-ni[1],mi[2]-ni[2],p)
		ℓi1 = k_inv_new_1d(mi1-ni1,p)
		ℓi2 = k_inv_new_1d(mi2-ni2,p)
		if ℓi1!= nothing && ℓi2 != nothing
			block = V_at_mi(ℓi1,ℓi2,V)
		else
			block = zeros(ComplexF64,4,4)
		end

		fill_block_m(Mi,Ni,H,block,p)
	end
	H = swap2(H)
	test_hermitianity(H,"V")
	H
	# Hermitian(H)
end


function test_fill_block(V,p)
	H = zeros(ComplexF64,p.Mfull,p.Mfull)
	Mi = [1,2]; Ni = [3,1]
	v = V_at_mi(1,2,V)
	fill_block_m(Mi,Ni,H,v,p)
	plot_heatmap(imag.(H)+real.(H),"test_fill")
end




function heatmap_of_coords(p) # heatmaps of m1-n1 and of m2-n2
	coord1y = [ilin2im(j,p)[1] for i=1:p.Mfull, j=1:p.Mfull]
	coord1x = [ilin2im(i,p)[1] for i=1:p.Mfull, j=1:p.Mfull]
	coord2y = [ilin2im(j,p)[2] for i=1:p.Mfull, j=1:p.Mfull]
	coord2x = [ilin2im(i,p)[2] for i=1:p.Mfull, j=1:p.Mfull]
	plot_heatmap(coord1y,"coord1")
	plot_heatmap(coord2y,"coord2")
	diffs_first1 = [coord1x[i,j]-coord1y[i,j] for i=1:p.Mfull, j=1:p.Mfull]
	diffs_first2 = [coord2x[i,j]-coord2y[i,j] for i=1:p.Mfull, j=1:p.Mfull]
	plot_heatmap(diffs_first2,"diffs_first2")
	plot_heatmap(diffs_first1,"diffs_first1")
end
##### New coordinates change

# from label of Fourier to label of the full matrix, gives the label of the left coefficient
im2ilin(m,p) = 1+4*((m[1]-1)*p.N+(m[2]-1))

function ilin2im(i,p) # inverse of im2ilin
	m1 = floor(Int,mod(i-1,4*p.N)/4) +1
	m2 = floor(Int,(i-1)/(4*p.N)) +1
	(m1,m2)
end

function test_it(p)
	init = [i for i=1:p.Mfull]
	deux = [ilin2im(i,p)[1] for i=1:p.Mfull]
	trois = [ilin2im(i,p)[2] for i=1:p.Mfull]
	px(init,"\n",deux,"\n",trois)
end



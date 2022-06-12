include("misc/lobpcg.jl")
include("common_functions.jl")
include("misc/create_bm_pot.jl")
# include("misc/plot.jl")
include("effective_potentials.jl")
using LinearAlgebra, JLD, FFTW, Plots, LaTeXStrings

mutable struct Basis
	# Parameters of the 2d cell
	a; a1; a2; a1_star; a2_star
	x_axis_cart
	dx; dS; dv
	k_axis; k_grid
	K_red
	N

	cell_area; Vol
	dim
	lattice_2d
	lattice_type_2πS3
	m_q1
	a1_micro; a2_micro
	a1_star_micro; a2_star_micro
	q1; q2; q3; q1_red; q2_red; q3_red
	R_four_2d
	M_four_2d
	vF
	sqi

	# Useless parameters
	Nz; L; dz
	kz_axis

	# Parameters for the 4×4 matrix
	N2; Mfull
	Li_tot; Li_k; Ci_tot; Ci_k; k2lin
	k_grid_lin
	S # part_hole matrix
	l; l1; l2; n_l
	ISΣ # S^{-1/2}
	solver
	Iα; Iβ
	fermi_velocity
	double_dirac # true if 4×4 matrix, false if 2×2 matrix
	coef_derivations # applies a coefficient to all derivation operator
	coef_energies_plot

	# Parameters for plots
	root_path
	energy_center; energy_scale
	folder_plots_bands
	folder_plots_matrices
	resolution_bands
	energy_unit_plots # ∈ ["eV","Hartree"]
	plots_article
	path_plots_article
	path_bandwidths
	function Basis()
		p = new()
		p.double_dirac = true
		p.coef_energies_plot = 1
		p.vF = 0.380
		p.plots_article = false
		p.path_plots_article = "../../bm/ab_initio_model/pics/"
		p
	end
end

function init_basis(p)
	### Parameters of the 2d cell
	p.k_axis = Int.(fftfreq(p.N)*p.N)
	p.root_path = "band_diagrams_bm_like/"
	create_dir(p.root_path)
	p.path_bandwidths = string(p.root_path,"bandwidths/")
	create_dir(p.path_bandwidths)

	p.folder_plots_matrices = "matrices/"
	path2 = string(p.root_path,p.folder_plots_matrices)
	create_dir(path2)
	init_cell_vectors(p;moire=true)
	p.S = nothing

	### Parameters for the 4×4 matrix
	number_of_states = p.double_dirac ? 4 : 2
	p.N2 = p.N^2; p.Mfull = number_of_states*p.N2

	ref = zeros(number_of_states,p.N,p.N); ref_k = zeros(p.N,p.N)
	p.Li_tot = LinearIndices(ref);    p.Li_k = LinearIndices(ref_k)
	p.Ci_tot = CartesianIndices(ref); p.Ci_k = CartesianIndices(ref_k)
	init_klist2(p)

	p.ISΣ = -1 # matrix (1+S_Σ)^(-1/2)
	p.solver = "LOBPCG"
	p.fermi_velocity = 0.38

	# Matrices to apply weights
	init_matrices_weights(p)

	# Eigenvalues
	p.l1 = floor(Int,p.Mfull/2) - p.l
	p.l2 = floor(Int,p.Mfull/2) + p.l + 1
	p.n_l = 2*p.l + 2

	p.k_grid_lin = [(0,0) for i=1:p.N, j=1:p.N]
	k = 1
	for i=1:p.N, j=1:p.N
		li = p.Li_k[i,j]
		p.k_grid_lin[li] = (p.k_axis[i],p.k_axis[j])
		k += 1
	end
end

reload_a(p) = init_cell_vectors(p;moire=true)

######################### Main functions, to plot band diagrams

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

coords_ik2full_i(mi1,mi2,p) = (p.Li_tot[1,mi1,mi2],p.Li_tot[2,mi1,mi2],p.Li_tot[3,mi1,mi2],p.Li_tot[4,mi1,mi2])


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


######################### New functions


X(n,I,p) = 2*(n-1)+(I-1)*2*p.N2+1

function fillM_Δ(H,K,i,I,j,J,V,a,b,p;star=false)
	x = X(i,I,p)
	y = X(j,J,p)

	jj = j
	n1,n2 = p.k_grid_lin[jj]
	KGj = K + [n1,n2]
	Qj = k_red2cart(KGj,p)
	n = norm(Qj)^2

	if !star
		H[x, y]     = n*V[1][a,b]
		H[x, y+1]   = n*V[2][a,b]
		H[x+1, y]   = n*V[3][a,b]
		H[x+1, y+1] = n*V[4][a,b]
	else
		H[y, x]     = n*conj(V[1][a,b])
		H[y+1,x]    = n*conj(V[2][a,b])
		H[y,x+1]    = n*conj(V[3][a,b])
		H[y+1,x+1]  = n*conj(V[4][a,b])
	end
end

function fillM(H,i,I,j,J,M,a,b,p;star=false)
	x = X(i,I,p)
	y = X(j,J,p)

	if !star
		H[x, y]     = M[1][a,b]
		H[x, y+1]   = M[2][a,b]
		H[x+1, y]   = M[3][a,b]
		H[x+1, y+1] = M[4][a,b]
	else
		H[y, x]     = conj(M[1][a,b])
		H[y+1,x]    = conj(M[2][a,b])
		H[y,x+1]    = conj(M[3][a,b])
		H[y+1,x+1]  = conj(M[4][a,b])
	end
end

function fillM_∇(H,K,i,I,j,J,A1,A2,a,b,p;star=false)
	x = X(i,I,p)
	y = X(j,J,p)

	jj = star ? i : j
	# jj = j
	n1,n2 = p.k_grid_lin[jj]
	KGj = K + [n1,n2]
	Qj = k_red2cart(KGj,p)

	if !star
		H[x,y]     = Qj[1]*A1[1][a,b]+Qj[2]*A2[1][a,b]
		H[x,y+1]   = Qj[1]*A1[2][a,b]+Qj[2]*A2[2][a,b]
		H[x+1,y]   = Qj[1]*A1[3][a,b]+Qj[2]*A2[3][a,b]
		H[x+1,y+1] = Qj[1]*A1[4][a,b]+Qj[2]*A2[4][a,b]
	else
		H[y,x]      = conj(Qj[1]*A1[1][a,b]+Qj[2]*A2[1][a,b])
		H[y+1,x]    = conj(Qj[1]*A1[2][a,b]+Qj[2]*A2[2][a,b])
		H[y,x+1]    = conj(Qj[1]*A1[3][a,b]+Qj[2]*A2[3][a,b])
		H[y+1,x+1]  = conj(Qj[1]*A1[4][a,b]+Qj[2]*A2[4][a,b])
	end
end

function fill_divΣ∇(H,Σ,K1,K2,i,I,j,J,a,b,p;star=false)
	x = X(i,I,p)
	y = X(j,J,p)

	n1,n2 = p.k_grid_lin[i]
	KGi = K2 + [n1,n2]
	Qi = k_red2cart(KGi,p)

	m1,m2 = p.k_grid_lin[j]
	KGj = K1 + [m1,m2]
	Qj = k_red2cart(KGj,p)

	n = Qi⋅Qj

	if !star
		H[x, y]     = n*Σ[1][a,b]
		H[x, y+1]   = n*Σ[2][a,b]
		H[x+1, y]   = n*Σ[3][a,b]
		H[x+1, y+1] = n*Σ[4][a,b]
	else
		H[y,x]      = n*conj(Σ[1][a,b])
		H[y+1,x]    = n*conj(Σ[2][a,b])
		H[y,x+1]    = n*conj(Σ[3][a,b])
		H[y+1,x+1]  = n*conj(Σ[4][a,b])
	end
end

function offdiag_div_Σ_∇(Σ,K,p;coef_∇=1,valley=1,K1=[0.0,0],K2=[0.0,0],name="",test=true)
	H = zeros(ComplexF64,4*p.N2, 4*p.N2)
	for i=1:p.N2
		n1,n2 = p.k_grid_lin[i]
		for j=1:p.N2
			m1,m2 = p.k_grid_lin[j]
			I1 = n1-m1; I2 = n2-m2
			c1,c2 = k_inv(I1,I2,p)
			if I1 in p.k_axis && I2 in p.k_axis
				fill_divΣ∇(H,Σ,K-K1,K-K2,i,1,j,2,c1,c2,p)
				fill_divΣ∇(H,Σ,K-K1,K-K2,i,1,j,2,c1,c2,p;star=true) # WHY NOT K-K1,K-K2 ?
			end
		end
	end
	# H = H+H'
	if test
		test_hermitianity(H,string(name,"div Σ ∇"))
	end
	H
end

function offdiag_A_k(A1,A2,K,p;coef_∇=1,valley=1,K1=[0.0,0],K2=[0.0,0],name="",test=true)
	H = zeros(ComplexF64,4*p.N2, 4*p.N2)
	for i=1:p.N2
		n1,n2 = p.k_grid_lin[i]
		for j=1:p.N2
			m1,m2 = p.k_grid_lin[j]
			I1 = n1-m1; I2 = n2-m2
			c1,c2 = k_inv(I1,I2,p)
			if I1 in p.k_axis && I2 in p.k_axis
				fillM_∇(H,K-K2,i,1,j,2,A1,A2,c1,c2,p)
				fillM_∇(H,K-K1,i,1,j,2,A1,A2,c1,c2,p;star=true)
				# fillM_∇(H,K-K2,i,1,j,2,A1,A2,c1,c2,p;star=true)
			end
		end
	end
	# H = H+H'
	if test
		test_hermitianity(H,string(name,"A"))
	end
	H
end

function σK(H,i,I,q,v,p;c=1,J=false)
	x = X(i,I,p)
	Q = k_red2cart(q,p)
	if !J
		H[x,x+1] = c*(v*Q[1] - im*Q[2])
		H[x+1,x] = c*(v*Q[1] + im*Q[2])
	else

		H[x,x+1] = c*(-v*Q[2] - im*Q[1])
		H[x+1,x] = c*(-v*Q[2] + im*Q[1])
	end
end

function mΔK(H,i,I,q,p)
	x = X(i,I,p)
	Q = k_red2cart(q,p)
	n = norm(Q)^2
	H[x,x] = n
	H[x+1,x+1] = n
end


# Creates [c1*σ(-i∇+k-K1)           0]
#         [0              σ(-i∇+k-K2)]
function Dirac_k(K,p;valley=1,K1=[0.0,0],K2=[0.0,0],coef_1=1,J=false,test=false)
	H = zeros(ComplexF64,4*p.N2, 4*p.N2)
	for i=1:p.N2
		n1,n2 = p.k_grid_lin[i]
		KG = K + [n1,n2]
		σK(H,i,1,KG-K1,valley,p;c=coef_1,J=J)
		σK(H,i,2,KG-K2,valley,p)
	end
	if test
		test_hermitianity(H,"σ(-i∇)")
	end
	H
end

function build_offdiag_V(V,p;test=false)
	H = zeros(ComplexF64,4*p.N2, 4*p.N2)
	for i=1:p.N2
		n1,n2 = p.k_grid_lin[i]
		for j=1:p.N2
			m1,m2 = p.k_grid_lin[j]
			I1 = n1-m1; I2 = n2-m2
			c1,c2 = k_inv(I1,I2,p)
			if I1 in p.k_axis && I2 in p.k_axis
				fillM(H,i,1,j,2,V,c1,c2,p)
				fillM(H,i,1,j,2,V,c1,c2,p;star=true)
			end
		end
	end
	if test
		test_hermitianity(H,"V")
	end
	H
end

function build_ondiag_W(Wplus,Wminus,p;test=true)
	H = zeros(ComplexF64,4*p.N2, 4*p.N2)
	for i=1:p.N2
		n1,n2 = p.k_grid_lin[i]
		for j=1:p.N2
			m1,m2 = p.k_grid_lin[j]
			c1,c2 = k_inv(n1-m1,n2-m2,p)
			fillM(H,i,1,j,1,Wplus ,c1,c2,p)
			fillM(H,i,2,j,2,Wminus,c1,c2,p)
		end
	end
	if test
		test_hermitianity(H,"W")
	end
	H
end


function ondiag_mΔ_k(K,p;K1=[0.0,0],K2=[0.0,0])
	H = zeros(ComplexF64,4*p.N2, 4*p.N2)
	for i=1:p.N2
		n1,n2 = p.k_grid_lin[i]
		KG = K + [n1,n2]
		mΔK(H,i,1,KG-K1,p)
		mΔK(H,i,2,KG-K2,p)
	end
	# test_hermitianity(H,"-Δ")
	H
end

function offdiag_mΔ_k(V,K,p;K1=[0.0,0],K2=[0.0,0],test=true)
	H = zeros(ComplexF64,4*p.N2, 4*p.N2)
	for i=1:p.N2
		n1,n2 = p.k_grid_lin[i]
		for j=1:p.N2
			m1,m2 = p.k_grid_lin[j]
			c1,c2 = k_inv(n1-m1,n2-m2,p)
			fillM_Δ(H,K-K2,i,1,j,2,V,c1,c2,p)
			fillM_Δ(H,K-K1,i,1,j,2,V,c1,c2,p;star=true)
		end
	end
	if test
		test_hermitianity(H,"-ΣΔ")
	end
	H
end




######################### Derivation operators

# Gives the action (hat{VX})_k in Fourier space. Vfour is the Fourier transform of V, Xfour is the Fourier transform of X
# actionV(Vfour,Xfour,p) = vcat(cyclic_conv(Vfour,Xfour)/length(Vfour)...)
# actionH(Vfour,p) = X -> p.k2lin.*X .+ actionV(Vfour,Reshape(X,p),p) # X = Xlin, v in direct space



interm(V,p) = apply_map_four(X -> [1 -1;0 1]*X,V,p)

# Creates (-i∇+k)^2 𝕀_{4×4}
function mΔ(k,p)
	n = p.Mfull
	Δ = zeros(ComplexF64,n,n)
	kC = k[1]*p.a1_star + k[2]*p.a2_star
	for ck_lin=1:p.N2
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
	for ck_lin=1:p.N2
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
				# c = conj(v[α-2,β][Ki1,Ki2])
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

function V_offdiag_matrix22(v0,p) # v = [v1,v2,v3,v4] = mat(v1 & v2 \\ v3 & v4), Fourier coeffcients
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
				# c = conj(v[α-2,β][Ki1,Ki2])
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


######################### Magnetic potential/derivation operators

# Creates [0          𝔸⋅(-i∇+k)]
#         [𝔸*⋅(-i∇+k)         0]

# Creates [0            V(-i∇+k)^2]
#         [V^*(-i∇+k)^2          0]

# Creates σ(-i∇+k)

# Creates (-i∇+k)^2 𝕀_2×2

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
function spectrum_on_a_path(Hv,Kdep,Klist,p;print_progress=false)
	res = p.resolution_bands
	n = length(Klist)
	n_path_points = res*n
	X = -1
	graphs = zeros(n_path_points,p.n_l)
	if print_progress px("Computation of the band diagram. Progress in % (multi-threaded) :") end
	for Ki=1:n
		K0 = Klist[Ki]; K1 = Klist[mod1(Ki+1,n)]
		path = [(1-t/res)*K0 .+ (t/res)*K1 for t=0:res-1]
		Threads.@threads for s=1:res
		# for s=1:res
			HvK = Hv + Kdep(path[s])
			# px("K ",K0," ",K1)
			# test_hermitianity(HvK,"Hvk")
			# test_part_hole_sym_matrix(HvK,p,"Hvk")
			h = herm(HvK)
			if h>1e-5 px("Be careful, H not exactly Hermitian : ",h) end
			HvK_herm = (HvK+HvK')/2
			(E,Xf) = solve_one(Hermitian(HvK_herm),p,X)
			# Selects eigenvalues around the Fermi level
			E = E[p.l1:p.l2]
			X = Xf
			indice = (Ki-1)*res + s
			graphs[indice,:] = E
			if print_progress # does not work because it's parallelized
				percentage = 100*((Ki-1)*res+s-1)/(n*res)
				print(percentage," ")
			end
		end
	end
	px("\n")
	graphs
end

fermi_label(p) = floor(Int,p.n_l/2) # the lower one, the upper one is fermi_label +1

function bandwidth(σ,p)
	nmid = fermi_label(p)
	diff = σ[:,nmid+1] .- σ[:,nmid]
	# px("Full diffs ",diff)
	# px("Verify spectrum ",σ[1,nmid-2]," ",σ[1,nmid-1]," ",σ[1,nmid]," ",σ[1,nmid+1]," ",σ[1,nmid+2])
	# px("Verify differences ",σ[1,nmid-1]-σ[1,nmid-2]," ",σ[1,nmid]-σ[1,nmid-1]," ",σ[1,nmid+1]-σ[1,nmid]," ",σ[1,nmid+2]-σ[1,nmid+1])
	# @assert diff[1] < 0.5*diff_verif[1] # verify that we take the right one
	maximum(σ[:,nmid+1])-minimum(σ[:,nmid])
end

function coef_plot_meV(θ,p)
	θrad = (π/180)*θ
	εθ = 2*sin.(θrad/2)
	p.coef_energies_plot = hartree_to_ev*1e3*εθ
end

function plot_bandwidths(θs,bw_bm,bw_ours,p;def_ticks=true)
	res_fig = 400
	f = CairoMakie.Figure(resolution=(res_fig+150,res_fig))
	ax = CairoMakie.Axis(f[1, 1], xlabel = "θ (degrees)", ylabel="Bandwidth (meV)")
	px("θs ",θs[1]," ",θs[end])
	if def_ticks ax.xticks = (θs[1]: 0.1 :θs[end]) end
	CairoMakie.xlims!(ax,θs[1],θs[end])

	colors = [:black,:red]
	n = length(θs)
	bws = [bw_bm,bw_ours]
	ymax = 0
	for j=1:2
		ys = [bws[j][i]*coef_plot_meV(θs[i],p) for i=1:n]
		if ymax < maximum(ys) ymax = maximum(ys) end
		points = [CairoMakie.Point2f0(θs[i],ys[i]) for i=1:n]
		CairoMakie.lines!(points,color=colors[j])
	end
	CairoMakie.ylims!(ax,0,ymax)


	paths = [p.path_bandwidths]
	if p.plots_article push!(paths,p.path_plots_article) end
	for path in paths
		CairoMakie.save(string(path,"bandwidths.pdf"),f)
	end
end

# w in meV, result in degrees
α2θ(α,w,p) = (180/π)*2*asin(w*1e-3*ev_to_hartree/(2*kD(p)*p.vF*α))

# From the numbers of the band diagram, produces a plot of it
function plot_band_diagram(σs,θs,Klist,Knames,name,p;K_relative=[0.0,0.0],shifts=zeros(100),energy_center=0,post_name="",colors=fill(:black,100))
	# Prepares the lists to plot
	n = length(Klist)
	res = p.resolution_bands
	n_path_points = res*n
	ylims = energy_center-p.energy_scale,energy_center+p.energy_scale
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

	# Builds figure
	res_fig = 700
	res_fig_x = 300
	f = CairoMakie.Figure(resolution=(res_fig_x,res_fig))
	ax = CairoMakie.Axis(f[1, 1], ylabel="meV")

	# CairoMakie.hidedecorations!(ax)
	# CairoMakie.hidexdecorations!(ax, ticks = false)
	# CairoMakie.hideydecorations!(ax, grid = false)

	# points_annotations = [CairoMakie.Point2f0(Kxs[i]+shift_text[1],Kys[i]+shift_text[1]) for i=1:n]
	# CairoMakie.annotations!(Knames, points_annotations,textsize=40)

	end_x = x_list[end]+dx_list[end] # fictitious x point, to loop the path
	CairoMakie.limits!(ax, x_list[1], end_x, ylims[1], ylims[2]) # x1, x2, y1, y2
	ax.yticks = (ylims[1] : 50 : ylims[2])

	# pl = Plots.plot(size=(1000,1100),ylims=ylims,legend=false) #:topright)
	for g=1:length(σs)
		for l=1:p.n_l
			s = shifts[g]
			points = [CairoMakie.Point2f0(x_list[i],(σs[g][i,l] + s)*coef_plot_meV(θs[g],p)) for i=1:n_path_points]
			points = vcat(points,[CairoMakie.Point2f0(end_x,(σs[g][1,l] + s)*coef_plot_meV(θs[g],p))])
			# Plots.plot!(pl,x_list,σs[g][:,l]*p.coef_energies_plot,xticks=nothing)
			CairoMakie.lines!(points,color=colors[g])
		end
	end


	# Vertical lines and ticks
	list_x_vert_labels = vcat([x_list[(i-1)*res+1] for i=1:n],[x_list[end]+dx_list[end]])
	m = length(list_x_vert_labels)
	labels = [LaTeXString(string("\$",Knames[mod1(i,n)],"\$")) for i=1:n+1]
	
	pos = [CairoMakie.Point2f0(list_x_vert_labels[i] + (i==m ? -10 : 0),ylims[1]-abs(ylims[1])*0.0) for i=1:m]
	CairoMakie.text!(labels, position=pos,textsize=20,font ="Arial bold")
	ax.xticks = (list_x_vert_labels,["" for i=1:m])

	# Saves
	s = string(name,"000000000")
	title = s[1:min(6,length(s))]
	path_local = string(p.root_path,p.folder_plots_bands,"/")
	create_dir(path_local)
	paths_plots = [path_local]

	if p.plots_article push!(paths_plots,p.path_plots_article) end
	for path in paths_plots
		ext = path == path_local ? "png" : "pdf"
		CairoMakie.save(string(path,"band_struct_",title,"_",post_name,".",ext),f)
	end
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

function plot_sequence_of_points(Klist,Knames,p;shifts_text=0,color=:black,linewidth=1,dots=true)
	n = length(Klist)
	Kcart = [k_red2cart(Klist[i],p) for i=1:n]
	Kxs = [Kcart[i][1] for i=1:n]
	Kys = [Kcart[i][2] for i=1:n]
	
	# Preparation 
	points = [CairoMakie.Point2f0(Kxs[i],Kys[i]) for i=1:n]

	# Lines
	loop = vcat(points,[points[1]])
	CairoMakie.lines!(loop,color=color,linewidth=linewidth)

	# Annotations
	st = shifts_text
	if shifts_text==0 st = [[0.0,0.0] for i=1:n] end
	points_annotations = [CairoMakie.Point2f0(Kxs[i]+st[i][1],Kys[i]+st[i][2]) for i=1:n]
	filtered_names = []; filt_pos = []
	for i=1:n
		s = Knames[i]
		if s != ""
			push!(filtered_names,s)
			push!(filt_pos,points_annotations[i])
		end
	end
	Knames_tex = [LaTeXString(string("\$",filtered_names[i],"\$")) for i=1:length(filtered_names)]
	CairoMakie.text!(Knames_tex, position=filt_pos,textsize=30)

	# Points
	if dots
		CairoMakie.scatter!(points,color=:black)
	end
end

plot_one_vector(v,name,p;shift_text=[0.0,0.0],linewidth=1,color=:black) = plot_sequence_of_points([[0.0,0],v],["",name],p;shifts_text=[[0.0,0.0],shift_text],linewidth=linewidth,color=color)

function plot_path(Klist,Knames,p)
	# Init
	res = 1000
	f = CairoMakie.Figure(resolution=(res,res))

	ax = CairoMakie.Axis(f[1, 1],aspect=1)
	ax.aspect = CairoMakie.AxisAspect(1)

	CairoMakie.hidedecorations!(ax)
	CairoMakie.hidexdecorations!(ax, grid  = false)
	CairoMakie.hideydecorations!(ax, ticks = false)

	lims = 4
	CairoMakie.limits!(ax, -lims, lims, -lims, lims) # x1, x2, y1, y2

	# q1, q2, q3
	q1 = [-1,-1]/3; q2 = [2,-1]/3; q3 = [-1,2]/3
	plot_one_vector(q1,"q_1",p;shift_text=[-0.2,-0.2],linewidth=3)
	plot_one_vector(q2,"q_2",p;shift_text=[0.1,-0.2],linewidth=3)
	plot_one_vector(q3,"q_3",p;shift_text=[0,0.1],linewidth=3)
	
	# Hexagons
	hexs = []
	hexag = [q3,q3+q2,q2,q2+q1,q1,q3+q1]
	push!(hexs,hexag)
	push!(hexs,[hexag[i]+[1,0] for i=1:length(hexag)])
	push!(hexs,[hexag[i]+[0,1] for i=1:length(hexag)])
	push!(hexs,[hexag[i]+[1,0] for i=1:length(hexag)])
	push!(hexs,[hexag[i]+[-1,1] for i=1:length(hexag)])
	push!(hexs,[hexag[i]+[-1,0] for i=1:length(hexag)])
	for hex in hexs
		plot_sequence_of_points(hex,["" for i=1:length(hex)],p;linewidth=1,dots=false,color=:grey)
	end

	# Plot list
	shifts_text = [[0.0,0.0] for i=1:length(Knames)]
	for i=1:length(Knames)
		if Knames[i]=="K_2" shifts_text[i] = [-0.3;-0.3] end
		if Knames[i]=="K_1" shifts_text[i] = [-0.3;0] end
		if Knames[i]=="Γ"   shifts_text[i] = [0.1;-0.2] end
		if Knames[i]=="M"   shifts_text[i] = [-0.2;0] end
		if Knames[i]=="Γ'"   shifts_text[i] = [-0.3;-0.1] end
	end
	plot_sequence_of_points(Klist,Knames,p;color=:blue,linewidth=5,shifts_text=shifts_text)

	# a1*,a2*
	plot_one_vector([1.0,0.0],"a_{1,M}^*",p;shift_text=[0.1,-0.1],color=:orange,linewidth=2)
	plot_one_vector([0.0,1.0],"a_{2,M}^*",p;shift_text=[0.1,-0.1],color=:orange,linewidth=2)

	# Saves
	path_local = "band_diagrams_bm_like/"
	paths_plots = [path_local]

	if p.plots_article push!(paths_plots,p.path_plots_article) end
	for path in paths_plots
		save(string(path,"path_bands_diagram.pdf"),f)
	end

	px("Path plotted")
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

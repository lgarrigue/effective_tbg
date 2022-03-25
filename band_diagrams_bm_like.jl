include("misc/lobpcg.jl")
include("common_functions.jl")
include("create_bm_pot.jl")
include("misc/plot.jl")
include("effective_potentials.jl")
using LinearAlgebra, JLD, FFTW

mutable struct Basis
	# Parameters of the 2d cell
	N; k_axis
	a; a1; a2; a1_star; a2_star
	K_dirac_red

	# Parameters for the 4×4 matrix
	σ1; σ2 # Pauli matrices
	N2d; Mfull
	Li_tot; Li_k; Ci_tot; Ci_k; k2lin
	H0
	S # part_hole matrix
	l; l1; l2; n_l
	Ssv
	solver
	Iα; Iβ

	# Parameters for plots
	root_path = "band_diagrams_bm_like/"
	energy_center; energy_scale
	folder
	function Basis()
		p = new()
		p
	end
end

function init_basis(p)
	### Parameters of the 2d cell
	p.K_dirac_red = [-1/3;1/3]
	p.k_axis = Int.(fftfreq(p.N)*p.N)

	init_cell_vectors(p)

	p.S = nothing

	### Parameters for the 4×4 matrix
	p.σ1 = Hermitian([0 1;1 0]); p.σ2 = Hermitian([0 -im;im 0])
	p.N2d = p.N^2; p.Mfull = 4*p.N2d

	ref = zeros(4,p.N,p.N); ref_k = zeros(p.N,p.N)
	p.Li_tot = LinearIndices(ref);    p.Li_k = LinearIndices(ref_k)
	p.Ci_tot = CartesianIndices(ref); p.Ci_k = CartesianIndices(ref_k)
	init_klist2(p)


	p.H0 = create_H0(p)
	p.Ssv = -1 # matrix (1+Σ)^(-1/2)
	p.solver = "LOBPCG"

	# Matrices to apply weights
	init_matrices_weights(p)

	# Eigenvalues
	p.l1 = floor(Int,p.Mfull/2) - p.l
	p.l2 = floor(Int,p.Mfull/2) + p.l
	p.n_l = 2*p.l + 1

end

######################### Main functions, to plot band diagrams

function plot_band_structure(Hv,name,p;resolution=3)
	K = [0.0,0.0]
	K0 = [2/3,1/3]
	Γ = K.-K0
	M0 = [1/2,0]
	M = K.-M0

	Klist = [Γ,K,M]
	Klist_names = ["Γ","K","M"]

	# @time (σ_on_path,pl) = spectrum_on_a_path(Klist,Klist_names,p,Hv,resolution)
	(σ_on_path,pl) = spectrum_on_a_path(Klist,Klist_names,p,Hv,resolution)
	path = string(p.root_path,p.folder)
	if !isdir(path) mkdir(path) end
	savefig(pl,string(path,"/band_struct_",name,".png"))
end

function do_one_value(Hv,K,p)
	(E,Xf) = solve_one(Hv,K,p)
	pl = plot_spectrum(E)
	savefig(pl,"spectrum_K.png")
end

######################### Coordinates change

function init_klist2(p)
	p.k2lin = []
	for c_lin=1:p.Mfull
		(i,pix,piy) = lin2coord(c_lin,p)
		push!(p.k2lin,p.k_axis[pix]^2 + p.k_axis[piy]^2)
	end
end

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
coords_ik2full_i(mi1,mi2,p) = (p.Li_tot[1,mi1,mi2],p.Li_tot[2,mi1,mi2],p.Li_tot[3,mi1,mi2],p.Li_tot[4,mi1,mi2])
k_inv(m,p) = Int(mod(m,p.N))+1

######################### Functions coordinates change

pot2four(V) = [fft(V[i,j]) for i=1:4, j=1:4]
four2pot(V) = [ifft(V[i,j]) for i=1:4, j=1:4]
shift_pot(V) = [fftshift(V[i,j]) for i=1:4, j=1:4]
four2dir(ψ_four) = [ifft(ψ_four[i]) for i=1:4]

# from the solution of LOBPCG ψ_lin[4*p.N^2] to 4 vectors containing the fourier transforms
function lin2four(ψ_lin,p) 
	ψ_four = init_vec(p)
	for lin=1:p.Mfull
		(i,pix,piy) = lin2coord(lin,p)
		ψ_four[i][pix,piy] = ψ_lin[lin]
	end
	ψ_four
end

# Gives the action (hat{VX})_k in Fourier space. Vfour is the Fourier transform of V, Xfour is the Fourier transform of X
# actionV(Vfour,Xfour,p) = vcat(cyclic_conv(Vfour,Xfour)/length(Vfour)...)
# actionH(Vfour,p) = X -> p.k2lin.*X .+ actionV(Vfour,Reshape(X,p),p) # X = Xlin, v in direct space

# Creates [-iσ∇    0]
#         [0    -iσ∇]
function create_H0(p)
	n = p.Mfull
	H0 = zeros(ComplexF64,n,n)
	for ck_lin=1:p.N2d
		(mi1,mi2) = lin_k2coord_ik(ck_lin,p)
		(c1,c2,c3,c4) = coords_ik2full_i(mi1,mi2,p)
		(m1,m2) = k_axis(mi1,mi2,p)
		vC = vec2C(m1*p.a1_star + m2*p.a2_star)# + kC
		H0[c1,c2] = conj(vC)
		H0[c2,c1] = vC
		H0[c3,c4] = conj(vC)
		H0[c4,c3] = vC
	end
	test_hermitianity(H0,"H0")
	# save_H(H0,"H0",p)
	(2π/p.a)*H0
end

# Creates [σk    0]
#         [0    σk]
function shift_κ(κ,p) # κ in reduced coordinates, κ_cart = κ_red_1 a1_star + κ_red_2 a2_star
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	κ1 = κ[1]; κ2 = κ[2]
	kC = vec2C(κ1*p.a1_star + κ2*p.a2_star)
	ckC = conj(kC)
	for ck_lin=1:p.N2d
		(mi1,mi2) = lin_k2coord_ik(ck_lin,p)
		(c1,c2,c3,c4) = coords_ik2full_i(mi1,mi2,p)
		H[c1,c2] = ckC; H[c2,c1] = kC; H[c3,c4] = ckC; H[c4,c3] = kC
	end
	(2π/p.a)*H
end

# Creates [0  𝕍]
#         [𝕍* 0]
function V_offdiag_matrix(v,p) # v = [v1,v2,v3,v4] = mat(v1 & v2 \\ v3 & v4), Fourier coeffcients
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	for n_lin=1:n
		(α,ni1,ni2) = lin2coord(n_lin,p)
		(n1,n2) = k_axis(ni1,ni2,p)
		for m_lin=1:n
			(β,mi1,mi2) = lin2coord(m_lin,p)
			(m1,m2) = k_axis(mi1,mi2,p)

			c = 0
			Pi1 = k_inv(n1-m1,p); Pi2 = k_inv(n2-m2,p)
			Ki1 = k_inv(-n1+m1,p); Ki2 = k_inv(-n2+m2,p)

			if α ≥ 3 && β ≤ 2
				c = conj(v[β,α-2][Ki1,Ki2])
			elseif α ≤ 2 && β ≥ 3
				c = v[α,β-2][Pi1,Pi2]
			end
			
			H[n_lin,m_lin] = c
		end
	end
	test_hermitianity(H,"V matrix")
	test_part_hole_sym_matrix(H,p,"H")
	# save_H(H,"potential_V",p)
	# display([H[mod1(x,p.Mfull),mod1(y,p.Mfull)] for x=1:30, y=1:30])
	Hermitian(H)
end

# Creates [𝕎  0]
#         [0  𝕎]
function V_ondiag_matrix(v,p)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	for n_lin=1:n
		(α,ni1,ni2) = lin2coord(n_lin,p)
		(n1,n2) = k_axis(ni1,ni2,p)
		for m_lin=1:n
			(β,mi1,mi2) = lin2coord(m_lin,p)
			(m1,m2) = k_axis(mi1,mi2,p)

			c = 0
			Pi1 = k_inv(n1-m1,p); Pi2 = k_inv(n2-m2,p)
			if α ≤ 2 && β ≤ 2
				c = v[α,β][Pi1,Pi2]
			elseif α ≥ 3 && β ≥ 3
				c = v[α-2,β-2][Pi1,Pi2]
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

# Creates [0     -i𝔸∇]
#         [-i𝔸*∇    0]
function A_offdiag_matrix(A1,A2,p)
	n = p.Mfull
	H = zeros(ComplexF64,n,n)
	for n_lin=1:n
		(α,ni1,ni2) = lin2coord(n_lin,p)
		(n1,n2) = k_axis(ni1,ni2,p)
		for m_lin=1:n
			(β,mi1,mi2) = lin2coord(m_lin,p)
			(m1,m2) = k_axis(mi1,mi2,p)

			c = 0
			Pi1 = k_inv(n1-m1,p); Pi2 = k_inv(n2-m2,p)
			Ki1 = k_inv(-n1+m1,p); Ki2 = k_inv(-n2+m2,p)

			if α ≥ 3 && β ≤ 2
				c = (m1*p.a1_star[1]+m2*p.a2_star[1])*conj(A1[β,α-2][Ki1,Ki2])
				+ (m1*p.a1_star[2]+m2*p.a2_star[2])*conj(A2[β,α-2][Ki1,Ki2])
			elseif α ≤ 2 && β ≥ 3
				c = (m1*p.a1_star[1]+m2*p.a2_star[1])*A1[α,β-2][Pi1,Pi2]
				+ (m1*p.a1_star[2]+m2*p.a2_star[2])*A2[α,β-2][Pi1,Pi2]
			end
			H[n_lin,m_lin] = c
		end
	end
	test_hermitianity(H,"A")
	test_part_hole_sym_matrix(H,p,"A")
	# save_H(H,"potential_V",p)
	# display([H[mod1(x,p.Mfull),mod1(y,p.Mfull)] for x=1:30, y=1:30])
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
	(λs,φs,cv,Xf) = solve_lobpcg(L,p.Mfull,l,p.k2lin;maxiter=maxiter,tol=tol,X0=X0,full_diag=true)
	ψs_four = [lin2four(φs[i],p) for i=1:l]
	(λs,ψs_four,cv,Xf)
end

# Solve H_K for one K
function solve_one(Hv,K,p,X0=-1) # l is the number of eigenvalues we want
	X = []
	if X0 == -1
		X = randn(ComplexF64,p.Mfull,p.l2)
	else
		X = X0
	end
	shiftK = shift_κ(K,p)
	if p.Ssv != -1
		shiftK = p.Ssv*shiftK*p.Ssv
	end
	Hk = Hermitian(Hv .+ shiftK)

	# test_part_hole_sym_matrix(Hk,p,"Hvk")
	# test_hermitianity(Hk,"Hvk")
	if p.solver=="LOBPCG"
		(E,φs,c,Xf) = apply_lobpcg(Hk,p.l2,p,X)
	else
		(E,φs) = eigen(Hk)
		E = E[1:p.l2]; φs = φs[:,1:p.l2] # Peut etre pas bon pour φs
	end
	(E,Xf)
end


######################### Computes the band diagram

function spectrum_on_a_path(Klist,names,p,Hv,res = 5) # plots spectrum for eigenvalues between l1 and l2
	n = length(Klist)
	T = [i/res for i=0:res-1]; Nt = res
	n_path_points = res*n
	n_eigenvals = 1
	graphs = zeros(n_path_points,n_eigenvals)
	Klistt = vcat(Klist,Klist[1])
	X = -1
	graphs = zeros(n_path_points,p.n_l)
	for Ki=1:n
		K0 = Klistt[Ki]; K1 = Klistt[Ki+1]
		path = [(1-t/Nt)*K0 .+ (t/Nt)*K1 for t=0:Nt-1]
		Threads.@threads for s=1:Nt
			(E,Xf) = solve_one(Hv,path[s],p,X)
			# Selects only positive eigenvalues
			E = E[p.l1:p.l2]
			X = Xf
			indice = (Ki-1)*res + s
			graphs[indice,:] = E
			percentage = 100*(Ki*(Nt-1)+s-1)/((n)*(Nt))
			# px("Percentage done ",percentage)
		end
	end
	ylims = p.energy_center-p.energy_scale,p.energy_center+p.energy_scale
	pl = plot(size=(1200,1300),legend=:topright,ylims=ylims)
	for l=1:p.n_l
		plot!(pl,graphs[:,l])
	end
	for Ki=1:n
		plot!(pl,[res*(Ki-1)+1], seriestype="vline", label=names[Ki])
	end
	# plot!(pl,graphs[:,1])
	# plot!(pl,[n_path_points+1], seriestype="vline", label=names[1])
	(graphs,pl)
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

			Pi1 = k_inv(n1+m1,p); Pi2 = k_inv(n2+m2,p)
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

######################### Low level functions

init_vec(p) = fill2d(zeros(ComplexF64,p.N,p.N),4)
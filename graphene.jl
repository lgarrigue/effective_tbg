include("common_functions.jl")
using DFTK, LinearAlgebra, FFTW, JLD
# setup_threading()
px = println

# + of functions
import Base.+  
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...)  

# Creates a dictionary which inverts an array, for instance [5,4,1] will give Dict(5 => 1, 2 => 4, 3 => 1)
inverse_dict_from_array(a) = Dict(value => key for (key, value) in Dict(zip((1:length(a)), a)))

mutable struct Params
	### Common parameters
	a; a1; a2; a1_star; a2_star
	x_axis_cart
	dx; dS; dz; dv
	k_axis; k_grid
	kz_axis
	K_red
	L # physical length of periodicity for the computations for a monolayer

	N; Nz; N2d; N3d
	n_fball # size of the Fermi ball ∈ ℕ
	x_axis_red; z_axis_red


	### Particular parameters

	M3d # M = (a1,a2), M3d = (a1,a2,L*e3) is the lattice

	# Dirac point quantities
	K_red_3d; K_coords_cart; K_kpt # Dirac point in several formats
	shift_K 

	# Monolayer functions
	i_state # index of first valence state
	u0; u1; u2 # in the Fourier ball
	u0_dir; u1_dir; u2_dir # in direct space
	v_monolayer_dir # in direct space
	v_0_M # hat{v_monolayer_dir}_{0,M}

	# Bilayer parameters
	interlayer_distance # physical distance between two layers /2

	# DFTK quantities
	ecut; scfres; basis
	kgrid
	Gvectors; Gvectors_cart; Gvectors_inv; Gplusk_vectors; Gplusk_vectors_cart # Gplusk is used to plot the e^{iKx} u(x) for instance
	recip_lattice; recip_lattice_inv # from k in reduced coords to a k in cartesian ones
	tol_scf

	# Misc
	ref_gauge # reference for fixing the phasis gauge freedom
	plots_cutoff
	root_path; path_exports; path_plots # paths
	function Params()
		p = new()
		p
	end
end

function init_params(p)
	init_cell_vectors(p)
	# Bilayer parameter

	p.M3d = [vcat(p.a1,[0]) vcat(p.a2,[0]) [0;0;p.L]]
	p.recip_lattice = DFTK.compute_recip_lattice(p.M3d)
	p.recip_lattice_inv = inv(p.recip_lattice)

	# Dirac point
	p.K_red_3d = vcat(p.K_red,[0])
	p.K_coords_cart = k_red2cart(p.K_red_3d,p)
	p.shift_K = [-1;0;0] # (1-R_{2π/3})K = -a_1^*, shift of this vector


	p.root_path = "graphene/"
	p.path_exports = string(p.root_path,"exported_functions/")
	p.path_plots = string(p.root_path,"plots/")
	create_dir(p.root_path)
	create_dir(p.path_exports)
	create_dir(p.path_plots)
	p
end

# from a list G=[a,b,c] of int, gives the iG such that Gvectors[iG]==G. If G is not in it, gives nothing
function index_of_Gvector(G,p) 
	tupleG = Tuple(G)
	if haskey(p.Gvectors_inv,tupleG)
		return p.Gvectors_inv[tupleG]
	else
		return nothing
	end
end

######################### Operations on functions

# Generates U_m := U_{L m}
# L is an action on the reduced Fourier space
function OpL(u, p, L) 
	Lu = zero(u)
	for iG=1:p.n_fball
		FiG = p.Gvectors[iG]
		LFiG = L(FiG)
		FinvLFiG = index_of_Gvector(LFiG,p)
		if isnothing(FinvLFiG)
			Lu[iG] = 0
		else
			Lu[iG] = u[FinvLFiG]
		end
	end
	Lu
end

function R(u,p) # (Ru)_G = u_{M G} or (Ru)^D_m = u^D_{F^{-1} M F(m)}. 
	M = [0 -1 0;1 -1 0; 0 0 1]
	L(G) = M*G
	OpL(u,p,L)
end

# equivalent to multiplication by e^{i cart(k) x), where k is in reduced
τ(u,k,p) = OpL(u,p,G -> G .- k)
# parity(u,p) = OpL(u,p,G -> -G)

z_translation(a,Z,p) = [a[x,y,mod1(z-Z,p.Nz)] for x=1:p.N, y=1:p.N, z=1:p.Nz]
r_translation(a,s,p) = [a[mod1(x-s[1],p.N),mod1(y-s[2],p.N),z] for x=1:p.N, y=1:p.N, z=1:p.Nz] # s ∈ {0,…,p.N-1}^2, 0 for no translation

######################### Solves Schrödinger's equations

# Obtain the Kohn-Sham potential
function scf_graphene_monolayer(p)
	C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
	c1 = [1/3,1/3,0.0]; c2 = -c1
	atoms = [C => [c1,c2]]
	model = model_PBE(p.M3d, atoms; temperature=1e-3, smearing=Smearing.Gaussian())
	basis = PlaneWaveBasis(model; Ecut=p.ecut, p.kgrid)

	(p.N,p.N,p.Nz) = basis.fft_size
	p.N3d = p.N^2*p.Nz
	p.x_axis_red = ((0:p.N-1))/p.N
	p.z_axis_red = ((0:p.Nz-1))/p.Nz

	init_cell_infinitesimals(p)
	@assert abs(p.dv - basis.dvol) < 1e-10

	# Run SCF
	p.scfres = self_consistent_field(basis)
	p.v_monolayer_dir = DFTK.total_local_potential(p.scfres.ham)[:,:,:,1]
	p.v_0_M = fft([sum(p.v_monolayer_dir[:,:,z])/p.N^2 for z=1:p.Nz])
end

# Compute the band structure at momentum k
function diag_monolayer_at_k(k,p;n_bands=10) # k is in reduced coordinates, n_bands is the number of eigenvalues we want
	K_dirac_coord = [k]
	# ksymops = [[DFTK.one(DFTK.SymOp)] for _ in 1:length(K_dirac_coord)]
	ksymops  = [[DFTK.identity_symop()] for _ in 1:length(K_dirac_coord)]
	basis = PlaneWaveBasis(p.scfres.basis, K_dirac_coord, ksymops)
	ham = Hamiltonian(basis; p.scfres.ρ)
	# Diagonalize H_K_dirac
	data = diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands + 3; n_conv_check=n_bands, tol=p.tol_scf, show_progress=true)
	if !data.converged
		@warn "Eigensolver not converged" iterations=data.iterations
	end
	# Extracts solutions
	sol = DFTK.select_eigenpairs_all_kblocks(data, 1:n_bands)
	Es = sol.λ[1]
	# px("Energies are ",Es)
	(Es,sol.X[1],basis,ham)
end

# Computes the Kohn-Sham potential of the bilayer at some stacking shift (disregistry)
function scf_graphene_bilayer(stacking_shift,p)
	sf = [0.0,0.0]
	sf[1] = p.x_axis_red[mod1(stacking_shift[1]+1,p.N)]
	sf[2] = p.x_axis_red[mod1(stacking_shift[2]+1,p.N)]
	total_shift = vcat(sf,[0.0])
	base = [1/3,1/3] # shift base
	s = -base
	D = p.interlayer_distance/(p.L*2)
	# px("D = ",D)
	c1_plus =  [1/3,1/3, D]; c2_plus = vcat(s,[D])
	c1_moins = [1/3,1/3,-D] .+ total_shift; c2_moins = vcat(s,[-D]) .+ total_shift
	C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
	atoms = [C => [c1_plus,c2_plus,c1_moins,c2_moins]]
	n_extra_states = 1

	model = model_PBE(p.M3d, atoms; temperature=1e-3, smearing=Smearing.Gaussian())
	basis = PlaneWaveBasis(model; Ecut=p.ecut, kgrid=p.kgrid)
	@assert (p.N,p.N,p.Nz) == basis.fft_size
	@assert abs(p.dv - basis.dvol) < 1e-10
	@assert p.x_axis_red == first.(DFTK.r_vectors(basis))[:,1,1]

	scfres = self_consistent_field(basis;tol=p.tol_scf,n_ep_extra=n_extra_states,eigensolver=lobpcg_hyper,maxiter=100,callback=x->nothing)
	Vks = DFTK.total_local_potential(scfres.ham)[:,:,:,1] # select first spin component
	Vks
end

# coords in reduced to coords in cartesian
k_red2cart(k_red,p) = p.recip_lattice*k_red
# coords in cartesian to coords in reduced
k_cart2red(k_cart,p) = p.recip_lattice_inv*k_cart

function get_dirac_eigenmodes(p)
	(Es_K,us_K,basis,ham) = diag_monolayer_at_k(p.K_red_3d,p)
	p.basis = basis
	p.K_kpt = basis.kpoints[1]
	K_norm_cart = 4π/(3*p.a)
	@assert abs(K_norm_cart - norm(p.K_kpt.coordinate_cart)) < 1e-10

	p.Gvectors = collect(G_vectors(p.basis, p.K_kpt))
	p.Gvectors_cart = collect(G_vectors_cart(p.basis, p.K_kpt))
	p.Gplusk_vectors = collect(Gplusk_vectors(p.basis, p.K_kpt))
	p.Gplusk_vectors_cart = collect(Gplusk_vectors_cart(p.basis, p.K_kpt))

	p.Gvectors_inv = inverse_dict_from_array(Tuple.(p.Gvectors)) # builds the inverse

	n_eigenmodes = length(Es_K)
	p.n_fball = size(us_K[:,1])[1] # size of the Fourier ball
	px("Size of Fourier ball: ",p.n_fball)
	px("Size of Fourier cube: ",p.N,"×",p.N,"×",p.Nz)

	# Fixes the gauges
	ref_vec = ones(p.N,p.N,p.Nz)
	p.ref_gauge = ref_vec/norm(ref_vec) 
	for i=1:n_eigenmodes
		ui = us_K[:,i]
		ui_dir = G_to_r(basis,p.K_kpt,ui)
		λ = sum(conj(ui_dir).*p.ref_gauge)
		c = λ/abs(λ)
		ui_dir *= c
		us_K[:,i] = r_to_G(basis,p.K_kpt,ui_dir)
	end

	# Extracts relevant states
	p.u0 = us_K[:,p.i_state-1]
	p.u0_dir = G_to_r(p.basis,p.K_kpt,p.u0)
	p.u1 = us_K[:,p.i_state]
	p.u2 = us_K[:,p.i_state+1]
	H_K_dirac = ham.blocks[1]; Hmat = Array(H_K_dirac)

	# Verifications
	# res = norm(H_K_dirac * p.u1 - dot(p.u1, H_K_dirac * p.u1) * p.u1)
	# println("Verification residual norm ",res," eigenvalues (p.u1,p.u2) ",real(dot(p.u1, H_K_dirac * p.u1)),",",real(dot(p.u2, H_K_dirac * p.u2))," shoul equal ",real(Es_K[p.i_state]),",",real(Es_K[p.i_state+1])," Norm p.u1 ",norm(p.u1))
	nothing
end


######################### Obtain the good orthonormal basis for the periodic Bloch functions at Dirac points u1 and u2

function rotate_u1_and_u2(p)
	τau = cis(2π/3)
	(Ru1,Tu1) = (R(p.u1,p),τ(p.u1,p.shift_K,p))
	(Ru2,Tu2) = (R(p.u2,p),τ(p.u2,p.shift_K,p))
	d1 = Ru1.-τau*Tu1; d2 = Ru2.-τau*Tu2
	
	c = (norm(d1))^2
	s = d1'*d2
	f = (c/abs(s))^2

	U = (s/abs(s))/(sqrt(1+f))*p.u1 + (1/sqrt(1+1/f))*p.u2
	V = (s/abs(s))/(sqrt(1+f))*p.u1 - (1/sqrt(1+1/f))*p.u2

	(RU,TU) = (R(U,p),τ(U,p.shift_K,p))
	(RV,TV) = (R(V,p),τ(V,p.shift_K,p))

	I = argmin([norm(RU.-τau *TU),norm(RV.-τau *TV)])
	p.u1 = I == 1 ? U : V
	p.u2 = conj.(p.u1) # conj ∘ parity is conj in Fourier

	p.u1_dir = G_to_r(p.basis,p.K_kpt,p.u1)
	p.u2_dir = G_to_r(p.basis,p.K_kpt,p.u2)
end

######################### Vint

# Long step, computation of a Kohn-Sham potential for each disregistry
function hat_V_bilayer_Xs(p) # hat(V)^{bil,Xs}_{0,M}
	V = zeros(ComplexF64,p.N,p.N,p.Nz)
	print("Step : ")
	grid = [[sx,sy] for sx=1:p.N, sy=1:p.N]
	# Threads.@threads for s=1:p.N^2
	Threads.@threads for s=1:p.N^2
		sm = grid[s]; sx = sm[1]; sy = sm[2]
		print(s," ")
		v = scf_graphene_bilayer([sx,sy],p)
		V[sx,sy,:] = fft([sum(v[:,:,z])/p.N^2 for z=1:p.Nz])
	end
	print("\n")
	V
end

function compute_Vint_Xs(V_bil_Xs,p) # in direct space, used just to verify that is does not depend too much on Xs
	Vint_Xs = zeros(ComplexF64,p.N,p.N,p.Nz)
	for sx=1:p.N
		for sy=1:p.N
			for z=1:p.Nz
				Vint_Xs[sx,sy,z] = sum([cis(M*(2π/p.L)*p.z_axis_red[z])*(V_bil_Xs[sx,sy,M] -2*p.v_0_M[M]*cos(M*(2π/p.L)*p.interlayer_distance/2)) for M=1:p.Nz])
			end
		end
	end
	Vint_Xs
end

function form_Vint_from_Vint_Xs(V_bil_Xs,p) # computes the Fourier transform
	mean_V_bil_Xs = [sum([V_bil_Xs[sx,sy,M] for sx=1:p.N, sy=1:p.N])/p.N^2 for M=1:p.Nz]
	Vint_f = zeros(ComplexF64,p.Nz)
	for M=1:p.Nz
		Vint_f[M] = mean_V_bil_Xs[M] - 2*p.v_0_M[M]*cos(M*(2π/p.L)*p.interlayer_distance/2)
	end
	Vint_f
end

function plot_VintXs_Vint(Vint_Xs,Vint_f,p)
	Vint_Xs_without_Z = [Vint_Xs[x,y,z] - sum(Vint_Xs[:,:,z])/p.N^2 for x=1:p.N, y=1:p.N, z=1:p.Nz]
	px("Vint dependency in XY ",sum(abs.(Vint_Xs_without_Z))/sum(abs.(Vint_Xs)))
	plr = plot(heatmap(real.(intZ(Vint_Xs,p))),plot(real.(ifft(Vint_f))))
	savefig(plr,string(p.path_plots,"Vint.png"))
end

function export_Vint(Vint_f,p)
	filename = string(p.path_exports,"N",p.N,"_Nz",p.Nz,"_Vint.jld")
	save(filename,"N",p.N,"Nz",p.Nz,"a",p.a,"L",p.L,"interlayer_distance",p.interlayer_distance,"Vint",Vint_f)
	px("Exported : Vint for N=",p.N,", Nz=",p.Nz)
end

######################### Computes the Fermi velocity

function form_∇_term(u,w,j,p) # <u,(-i∂_j +K_j)w>
	GpKj = [p.Gplusk_vectors_cart[iG][j] for iG=1:p.n_fball]
	sum((conj.(u)) .* (GpKj.*w))
end

# 2×2 matrices <u,(-i∂_j +K_j)u>
form_∇_one_matrix(u1,u2,j,p) = Hermitian([form_∇_term(u1,u1,j,p) form_∇_term(u1,u2,j,p);form_∇_term(u2,u1,j,p) form_∇_term(u2,u2,j,p)])

function fermi_velocity_from_rotated_us(p) # needs that u1 and u2 are rotated before
	(A1,A2) = (form_∇_one_matrix(p.u1,p.u2,1,p),form_∇_one_matrix(p.u1,p.u2,2,p))
	# display(A1); display(A2)
	px("Fermi velocity ",abs(A1[1,2]))
end

function get_fermi_velocity_with_finite_diffs(p)
	n_samplings = 2*10+1 # has to be even
	Dλ = 0.001
	start_λ = 1-Dλ; end_λ = 1+Dλ; dλ = (end_λ-start_λ)/(n_samplings-1)
	set_coefs = [start_λ + i*dλ for i=0:n_samplings-1]
	set_cart_K = [λ*p.K_coords_cart for λ in set_coefs]
	values_down = []; values_up = []
	for i=1:n_samplings
		k = k_cart2red(set_cart_K[i],p)
		Es = (diag_monolayer_at_k(k,p))[1]
		push!(values_down,Es[p.i_state])
		push!(values_up,Es[p.i_state+1])
	end
	Ipoint = Int(floor(n_samplings/4)-1)
	dK = norm(set_cart_K[Ipoint+1]-set_cart_K[Ipoint])
	x_axis = norm.(set_cart_K)
	pl = plot(x_axis,[values_down,values_up]); display(pl)
	vF = (values_down[Ipoint+1]-values_down[Ipoint])/dK
	vF_up = (values_up[Ipoint+1]-values_up[Ipoint])/dK
	px("Fermi velocity: ",vF,". Verification with upper eigenvalue: ",vF_up)
end

function compute_vF_with_diagonalization_doesnt_work(p)
	(A1,A2) = (form_∇_one_matrix(p.u1,p.u2,1,p),form_∇_one_matrix(p.u1,p.u2,2,p))
	(E1,W1) = eigen(A1)
	(E2,W2) = eigen(A2)
	px("Eigenvals1 ",E1)
	px("Eigenvals2 ",E2)
	σ1 = Hermitian([0 1;1 0]); σ2 = Hermitian([0 -im;im 0])
	(Eσ1,V1) = eigen(σ1)
	(Eσ2,V2) = eigen(σ2)
	px(Eσ1," ",Eσ2)
	px("Matrices de passage qui devraient être égales")
	U1 = V1*W1; U2 = V2*W2

	function turn(u1,u2,U)
		α1 = conj(U[1,1])*u1 + conj(U[1,2])*u2
		α2 = conj(U[2,1])*u1 + conj(U[2,2])*u2
		(α1,α2)
	end
	(α1,α2) = turn(p.u1,p.u2,U1)
	Aα1 = form_∇_one_matrix(α1,α2,1,p)
	px("Aα1")
	display(Aα1)

	(β1,β2) = turn(p.u1,p.u2,U2)
	Aβ2 = form_∇_one_matrix(β1,β2,2,p)
	px("Aβ2")
	display(Aβ2)

	px("Egalite: ",norm(U1.-U2))

	id = [1 0;0 1]
	px("Verifs id ",norm(W1'*W1 .- id)," ",norm(V1'*V1 .- id))
end

######################### Exports

function exports_v_u1_u2(p)
	v = p.v_monolayer_dir
	v_f = fft(v)
	vu1_f = fft(v.*p.u1_dir)
	vu2_f = fft(v.*p.u2_dir)
	u1_f = fft(p.u1_dir)
	u2_f = fft(p.u2_dir)
	prods_f = [fft(abs2.(p.u1_dir)), fft(conj.(p.u1_dir).*p.u2_dir), fft(conj.(p.u2_dir).*p.u1_dir), fft(abs2.(p.u2_dir))]

	filename = string(p.path_exports,"N",p.N,"_Nz",p.Nz,"_u1_u2_V.jld")
	save(filename,"N",p.N,"Nz",p.Nz,"a",p.a,"L",p.L,"v_f",v_f,"vu1_f",vu1_f,"vu2_f",vu2_f,"u1_f",u1_f,"u2_f",u2_f,"prods_f",prods_f)
	px("Exported : V, u1, u2 functions for N=",p.N,", Nz=",p.Nz)
end

######################### Test symmetries

function test_rot_sym(p)
	# Tests
	(RS,TS) = (R(p.u1,p),τ(p.u1,p.shift_K,p))
	(RW,TW) = (R(p.u2,p),τ(p.u2,p.shift_K,p))
	(Ru0,Tu0) = (R(p.u0,p),τ(p.u0,p.shift_K,p))

	τau = cis(2π/3)
	px("Test R φ0 = φ0 ",norm(Ru0.-Tu0)/norm(Ru0))
	px("Test R φ1 = τ   φ1 ",norm(RS.-τau*TS)/norm(RS))
	px("Test R φ2 = τ^2 φ2 ",norm(RW.-τau^2*TW)/norm(RW))
end

######################### Plot functions

# fun in cart coords to fun in red coords
function arr2fun(u,p;bloch_trsf=true) # u in the Fourier ball, to function in cartesian
	Gs = bloch_trsf ? p.Gplusk_vectors_cart : p.Gvectors_cart # EQUIVALENT TO APPLY e^{iKx} !
	f(x,y,z) = 0
	for iG=1:length(Gs)
		g(x,y,z) = u[iG]*cis(Gs[iG]⋅[x,y,z])
		if norm(p.Gvectors[iG]) < p.plots_cutoff
			f = f + g
		end
	end
	f
end

function simple_plot(u,fun,Z,p;n_motifs=3,bloch_trsf=true,res=25)
	f = arr2fun(u,p;bloch_trsf=bloch_trsf)
	g = scale_fun3d(f,n_motifs)
	a = fun.([g(i/res,j/res,Z) for i=0:res-1, j=0:res-1])
	heatmap(a,size=(1000,1000))
end

function rapid_plot(u,p;n_motifs=5,name="rapidplot",bloch_trsf=true,res=25)
	Z = 0
	funs = [abs,real,imag]
	hm = [simple_plot(u,fun,Z,p;n_motifs=n_motifs,bloch_trsf=bloch_trsf,res=res) for fun in funs]
	size = 600
	r = length(funs)
	pl = plot(hm...,layout=(1,r),size=(r*size,size-200),legend=false)
	savefig(pl,string(p.path_plots,name,".png"))
end

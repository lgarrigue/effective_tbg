include("common_functions.jl")
using DFTK, LinearAlgebra, FFTW, JLD, LaTeXStrings
using SharedArrays, Distributed
using Plots
setup_threading()
px = println

mutable struct Params
	### Common parameters
	a; a1; a2; a1_star; a2_star
	x_axis_cart
	dx; dS; dz; dv
	k_axis; k_grid
	kz_axis
	K_red
	L # physical length of periodicity for the computations for a monolayer
	cell_area
	Vol # Volume Ω×L of the direct lattice
	Vol_recip # Volume of the reciprocal lattice
	sqrtVol

	N; Nz; N2d; N3d
	n_fball # size of the Fermi ball ∈ ℕ
	x_axis_red; z_axis_red; z_axis_cart

	### Particular parameters
	M3d # M = (a1,a2), M3d = (a1,a2,L*e3) is the lattice

	# Dirac point quantities
	K_red_3d; K_coords_cart; K_kpt # Dirac point in several formats
	shift_K 
	v_fermi # Fermi velocity

	### Monolayer functions
	i_state # index of first valence state
	# Dirac Bloch eigenfunctions
	u0_fb; u1_fb; u2_fb # in the Fourier ball
	u0_fc; u1_fc; u2_fc # in the Fourier cube
	u0_dir; u1_dir; u2_dir # in direct space
	# Kohn-Sham potential
	v_monolayer_dir # in direct space
	v_monolayer_fc # in the Fourier cube
	v_monolayer_fb # in the Fourier cube
	shifts_atoms # coordinates in 3d of the shifts of the two atoms of the unitthe two atoms of the unit cell
	# Non local pseudopotentials
	non_local_coef
	non_local_φ1_fb; non_local_φ1_fc
	non_local_φ2_fb; non_local_φ2_fc

	### Bilayer quantities
	interlayer_distance # physical distance between two layers
	Vint_f
	Vint_dir
	V_bilayer_Xs_fc
	Nint # the resolution for the disregistry lattice used to compute Vint

	# DFTK quantities
	ecut; scfres; basis; atoms; psp
	kgrid
	Gvectors; Gvectors_cart; Gvectors_inv; Gplusk_vectors; Gplusk_vectors_cart # Gplusk is used to plot the e^{iKx} u(x) for instance
	Gvectors_cart_fc # in the Fourier cube
	recip_lattice; recip_lattice_inv # from k in reduced coords to a k in cartesian ones
	tol_scf


	# Misc
	ref_gauge # reference for fixing the phasis gauge freedom
	plots_cutoff
	root_path; path_exports; path_plots # paths
	export_plots_article
	path_plots_article
	function Params()
		p = new()
		p
	end
end

norm_K_cart(a) = 4*π/(3*a)

function init_params(p)
	init_cell_vectors(p;rotate=true)
	# Bilayer parameter
	p.M3d = [vcat(p.a1,[0]) vcat(p.a2,[0]) [0;0;p.L]]
	p.Vol = DFTK.compute_unit_cell_volume(p.M3d)
	p.Vol_recip = (2π)^3/p.Vol
	p.recip_lattice = DFTK.compute_recip_lattice(p.M3d)
	p.recip_lattice_inv = inv(p.recip_lattice)

	# Dirac point
	p.K_red_3d = vcat(p.K_red,[0])
	p.K_coords_cart = k_red2cart_3d(p.K_red_3d,p)
	p.shift_K = [-1;0;0] # (1-R_{2π/3})K = -a_1^*, shift of this vector

	# Paths
	p.root_path = "graphene/"
	p.path_exports = string(p.root_path,"exported_functions/")
	p.path_plots = string(p.root_path,"plots/")
	p.export_plots_article = false
	p.path_plots_article = "../../bm/ab_initio_model/pics/"
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

# Generates V_m := U_{L m}
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

# (Ru)_G = u_{M G} or (Ru)^D_m = u^D_{F^{-1} M F(m)}
function R(u,p) 
	M = [0 -1 0;1 -1 0; 0 0 1]
	L(G) = M*G
	OpL(u,p,L)
end

function M(u,p) 
	M = [0 -1 0;-1 0 0; 0 0 -1]
	L(G) = M*G
	OpL(u,p,L)
end

# Equivalent to multiplication by e^{i cart(k) x), where k is in reduced, e^{im0 a^*⋅x} u = ∑ e^{ima^*⋅x} u_{m-m0}
τ(u,k,p) = OpL(u,p,G -> G .- k)
# Parity(u,p) = OpL(u,p,G -> -G)

z_translation(a,Z,p) = [a[x,y,mod1(z-Z,p.Nz)] for x=1:p.N, y=1:p.N, z=1:p.Nz]
r_translation(a,s,p) = [a[mod1(x-s[1],p.N),mod1(y-s[2],p.N),z] for x=1:p.N, y=1:p.N, z=1:p.Nz] # s ∈ {0,…,p.N-1}^2, 0 for no translation

######################### Solves Schrödinger's equations

# Obtain the Kohn-Sham potential
function scf_graphene_monolayer(p)
	p.psp = load_psp("hgh/pbe/c-q4")
	C = ElementPsp(:C, psp=p.psp)
	c1 = [1/3,1/3,0.0]; c2 = -c1
	p.shifts_atoms = [c1,c2]
	p.atoms = [C => [c1,c2]]
	model = model_PBE(p.M3d, p.atoms; temperature=1e-3, smearing=Smearing.Gaussian())
	basis = PlaneWaveBasis(model; Ecut=p.ecut, p.kgrid)
	# px("Number of spin components ",model.n_spin_components)

	(p.N,p.N,p.Nz) = basis.fft_size
	p.path_plots = string(p.root_path,"plots_N",p.N,"_Nz",p.Nz,"/")
	create_dir(p.path_plots)
	
	p.N3d = p.N^2*p.Nz
	p.x_axis_red = ((0:p.N -1))/p.N
	p.z_axis_red = ((0:p.Nz-1))/p.Nz
	p.z_axis_cart = p.z_axis_red*p.L

	init_cell_infinitesimals(p)
	@assert abs(p.dv - basis.dvol) < 1e-10

	# Extracts kpoint
	i_kpt = 1
	kpt = basis.kpoints[i_kpt]

	# Run SCF
	p.scfres = self_consistent_field(basis)
	px("Energies")
	display(p.scfres.energies)
	p.v_monolayer_dir = DFTK.total_local_potential(p.scfres.ham)[:,:,:,1]
	p.v_monolayer_fc = myfft(p.v_monolayer_dir,p.Vol)
	# p.v_monolayer_fb = r_to_G(basis,kpt,p.v_monolayer_dir)

	# Test potential energy
	potential_energy(u_dir,p) = sum(abs2.(u_dir).*p.v_monolayer_dir)*p.dv
	V_energy = 0
	# for i=1:length(basis.kpoints)
		# px("i ",i," ",basis.kpoints[i].coordinate)
	# end
	occupation = floor(Int,sum(p.scfres.ρ)*p.dv +0.5)
	px("Occupation ",occupation)
	for i=1:occupation
		ui = p.scfres.ψ[i_kpt][:, i]
		ui_dir = G_to_r(basis,kpt,ui)
		V_energy += potential_energy(ui_dir,p)
	end
	px("Potential energy 'by hand': ",V_energy," in Kpt ",kpt.coordinate)
end

# Compute the band structure at momentum k
function diag_monolayer_at_k(k,p;n_bands=10) # k is in reduced coordinates, n_bands is the number of eigenvalues we want
	# ksymops = [[DFTK.one(DFTK.SymOp)] for _ in 1:length(K_dirac_coord)]
	# ksymops  = [[DFTK.identity_symop()] for _ in 1:length(K_dirac_coord)]
	basis = PlaneWaveBasis(p.scfres.basis, [k], [1])
	ham = Hamiltonian(basis; p.scfres.ρ)
	# Diagonalize H_K_dirac
	data = diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands + 3; n_conv_check=n_bands, tol=p.tol_scf, show_progress=true)
	if !data.converged
		@warn "Eigensolver not converged" iterations=data.iterations
	end
	# Extracts solutions
	sol = DFTK.select_eigenpairs_all_kblocks(data, 1:n_bands)
	Es = sol.λ[1]
	us = [sol.X[1][:,i] for i=1:size(sol.X[1],2)]
	# px("Energies are ",Es)
	(Es,us,basis,ham)
end

# Computes the Kohn-Sham potential of the bilayer at some stacking shift (disregistry)
function scf_graphene_bilayer(sx,sy,p)
	stacking_shift_red = [sx,sy,0.0]
	D = p.interlayer_distance/(p.L*2)
	c1_plus =  [ 1/3, 1/3, D] .+ stacking_shift_red
	c2_plus =  [-1/3,-1/3, D] .+ stacking_shift_red
	c1_moins = [ 1/3, 1/3,-D]
	c2_moins = [-1/3,-1/3,-D]
	C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
	atoms = [C => [c1_plus,c2_plus,c1_moins,c2_moins]]
	n_extra_states = 1

	model = model_PBE(p.M3d, atoms; temperature=1e-3, smearing=Smearing.Gaussian())
	basis = PlaneWaveBasis(model; Ecut=p.ecut, kgrid=p.kgrid)
	@assert (p.N,p.N,p.Nz) == basis.fft_size
	@assert abs(p.dv - basis.dvol) < 1e-10
	@assert p.x_axis_red == first.(DFTK.r_vectors(basis))[:,1,1]

	scfres = self_consistent_field(basis;tol=p.tol_scf,n_ep_extra=n_extra_states,eigensolver=lobpcg_hyper,maxiter=100,callback=x->nothing)
	Vks_dir = DFTK.total_local_potential(scfres.ham)[:,:,:,1] # select first spin component
	myfft(Vks_dir,p.Vol)
end

# coords in reduced to coords in cartesian
k_red2cart_3d(k_red,p) = p.recip_lattice*k_red
# coords in cartesian to coords in reduced
k_cart2red(k_cart,p) = p.recip_lattice_inv*k_cart

# Obtains u0, u1 and u2 at the Dirac point
function get_dirac_eigenmodes(p)
	(Es_K,us_K,basis,ham) = diag_monolayer_at_k(p.K_red_3d,p)
	p.basis = basis
	p.K_kpt = basis.kpoints[1]
	K_norm_cart = 4π/(3*p.a)
	@assert abs(K_norm_cart - norm(k_red2cart_3d(p.K_kpt.coordinate,p))) < 1e-10

	p.Gvectors = collect(G_vectors(p.basis, p.K_kpt))
	p.Gvectors_cart = collect(G_vectors_cart(p.basis, p.K_kpt))
	p.Gplusk_vectors = collect(Gplusk_vectors(p.basis, p.K_kpt))
	p.Gplusk_vectors_cart = collect(Gplusk_vectors_cart(p.basis, p.K_kpt))

	p.Gvectors_inv = inverse_dict_from_array(Tuple.(p.Gvectors)) # builds the inverse


	n_eigenmodes = length(Es_K)
	p.n_fball = length(us_K[1]) # size of the Fourier ball
	px("Size of Fourier ball: ",p.n_fball)
	px("Size of Fourier cube: ",p.N,"×",p.N,"×",p.Nz)

	# Fixes the gauges
	ref_vec = ones(p.N,p.N,p.Nz)
	p.ref_gauge = ref_vec/norm(ref_vec) 
	for i=1:n_eigenmodes
		ui = us_K[i]
		ui_dir = G_to_r(basis,p.K_kpt,ui)
		λ = sum(conj(ui_dir).*p.ref_gauge)
		c = λ/abs(λ)
		ui_dir *= c
		us_K[i] = r_to_G(basis,p.K_kpt,ui_dir)
	end

	# Extracts relevant states
	p.u0_fb = us_K[p.i_state-1]
	p.u0_dir = G_to_r(p.basis,p.K_kpt,p.u0_fb)
	p.u0_fc = myfft(p.u0_dir,p.Vol)

	p.u1_fb = us_K[p.i_state]
	p.u2_fb = us_K[p.i_state+1]
	p.u1_dir = G_to_r(p.basis,p.K_kpt,p.u1_fb)
	p.u2_dir = G_to_r(p.basis,p.K_kpt,p.u2_fb)
	p.u1_fc = myfft(p.u1_dir,p.Vol)
	p.u2_fc = myfft(p.u2_dir,p.Vol)

	H_K_dirac = ham.blocks[1]; Hmat = Array(H_K_dirac)

	# Verifications
	# res = norm(H_K_dirac * p.u1_fb - dot(p.u1_fb, H_K_dirac * p.u1_fb) * p.u1_fb)
	# println("Verification residual norm ",res," eigenvalues (p.u1_fb,p.u2_fb) ",real(dot(p.u1_fb, H_K_dirac * p.u1_fb)),",",real(dot(p.u2_fb, H_K_dirac * p.u2_fb))," shoul equal ",real(Es_K[p.i_state]),",",real(Es_K[p.i_state+1])," Norm p.u1_fb ",norm(p.u1_fb))
	nothing
end


######################### Obtain the good orthonormal basis for the periodic Bloch functions at Dirac points u1 and u2

function rotate_u1_and_u2(p)
	τau = cis(2π/3)
	(Ru1,Tu1) = (R(p.u1_fb,p),τ(p.u1_fb,p.shift_K,p))
	(Ru2,Tu2) = (R(p.u2_fb,p),τ(p.u2_fb,p.shift_K,p))
	d1 = Ru1.-τau*Tu1; d2 = Ru2.-τau*Tu2
	
	c = (norm(d1))^2
	s = d1'*d2
	f = (c/abs(s))^2

	U = (s/abs(s))/(sqrt(1+f))*p.u1_fb + (1/sqrt(1+1/f))*p.u2_fb
	V = (s/abs(s))/(sqrt(1+f))*p.u1_fb - (1/sqrt(1+1/f))*p.u2_fb

	(RU,TU) = (R(U,p),τ(U,p.shift_K,p))
	(RV,TV) = (R(V,p),τ(V,p.shift_K,p))

	I = argmin([norm(RU.-τau *TU),norm(RV.-τau *TV)])
	p.u1_fb = I == 1 ? U : V
	# ϕ2(x) = conj(ϕ1(-x)) ⟹ u2(x) = conj(u1(-x))
	# # conj ∘ parity is conj in Fourier
	p.u2_fb = conj.(p.u1_fb)

	# Computes them in direct space
	p.u1_dir = G_to_r(p.basis,p.K_kpt,p.u1_fb)
	p.u2_dir = G_to_r(p.basis,p.K_kpt,p.u2_fb)
	# Computes them in the Fourier cube
	p.u1_fc = myfft(p.u1_dir,p.Vol)
	p.u2_fc = myfft(p.u2_dir,p.Vol)
	# px("LA ",norm(p.u0_fb)," ",norm(p.u1_fb)," ",norm(p.u0_dir)," ",norm(p.u1_dir)," ",norm(p.u0_fc)," ",norm(p.u1_fc))
end

######################### Computation of Vint

# Long step, computation of a Kohn-Sham potential for each disregistry
function compute_V_bilayer_Xs(p) # hat(V)^{bilayer,s}_{0,M}
	V = zeros(ComplexF64,p.Nint,p.Nint,p.Nz)
	print("Step : ")
	for six=1:p.Nint
		print(six," ")
		for siy=1:p.Nint
			(sx,sy) = ((six-1)/p.Nint,(siy-1)/p.Nint)
			v_fc = scf_graphene_bilayer(sx,sy,p)
			V[six,siy,:] = v_fc[1,1,:]
		end
	end
	print("\n")
	p.V_bilayer_Xs_fc = V
end

# Computes Vint_Xs
function compute_Vint_Xs(p)
	app(mz) = 2*cos(mz*π*p.interlayer_distance/p.L)
	V_app_k = p.v_monolayer_fc[1,1,:].*app.(p.kz_axis)
	[p.V_bilayer_Xs_fc[six,siy,miz] - V_app_k[miz] for six=1:p.Nint, siy=1:p.Nint, miz=1:p.Nz]/sqrt(p.cell_area)
end

# Computes Vint
form_Vint_from_Vint_Xs(Vint_Xs_fc,p) = [sum(Vint_Xs_fc[:,:,miz]) for miz=1:p.Nz]/p.Nint^2

# Computes the dependency of Vint_Xs on Xs
function computes_δ_Vint(Vint_Xs_fc,Vint_f,p)
	c = 0
	for sx=1:p.Nint, sy=1:p.Nint, mz=1:p.Nz
		c += abs2(Vint_Xs_fc[sx,sy,mz] - Vint_f[mz])
	end
	px("Dependency of Vint_Xs on Xs, δ_Vint= ",c/(p.Nint^2*sum(abs2.(Vint_f))))
end

######################### Computes the Fermi velocity

function form_∇_term(u,w,j,p) # <u,(-i∂_j +K_j)w>
	GpKj = [p.Gplusk_vectors_cart[iG][j] for iG=1:p.n_fball]
	sum((conj.(u)) .* (GpKj.*w))
end

# 2×2 matrices <u,(-i∂_j +K_j)u>
form_∇_one_matrix(u1,u2,j,p) = Hermitian([form_∇_term(u1,u1,j,p) form_∇_term(u1,u2,j,p);form_∇_term(u2,u1,j,p) form_∇_term(u2,u2,j,p)])

function fermi_velocity_from_rotated_us(p) # needs that u1 and u2 are rotated before
	(A1,A2) = (form_∇_one_matrix(p.u1_fb,p.u2_fb,1,p),form_∇_one_matrix(p.u1_fb,p.u2_fb,2,p))
	# display(A1); display(A2)
	px("Fermi velocity ",abs(A1[1,2]))
end

function get_fermi_velocity_with_finite_diffs(n_samplings_over_2,p)
	@assert n_samplings_over_2 ≥ 4
	n_samplings = 2*n_samplings_over_2+1 # has to be even
	Dλ = 0.0001
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
	pl = plot(x_axis,[values_down,values_up])
	savefig(pl,string(p.path_plots,"graph_fermi_velocity.png"))
	vF = (values_down[Ipoint+1]-values_down[Ipoint])/dK
	vF_up = (values_up[Ipoint+1]-values_up[Ipoint])/dK
	px("Fermi velocity: ",vF,". Verification with upper eigenvalue: ",vF_up)
	vF
end

function change_gauge_functions(ξ,p)
	α = cis(ξ); β = cis(-ξ)
	p.u0_fb *= α; p.u1_fb *= α; p.u2_fb *= β
	p.u0_fc *= α; p.u1_fc *= α; p.u2_fc *= β
	p.u0_dir *= α; p.u1_dir *= α; p.u2_dir *= β
	p.non_local_φ1_fb *= α; p.non_local_φ1_fc *= α
	p.non_local_φ2_fb *= β; p.non_local_φ2_fc *= β
end

function exchange_u1_u2(p)
	p.u1_fb,p.u2_fb = p.u2_fb,p.u1_fb 
	p.u1_fc,p.u2_fc = p.u2_fc,p.u1_fc
	p.u1_dir,p.u2_dir = p.u2_dir,p.u1_dir
end

function compute_scaprod_Dirac(p)
	(∂1_u2_f,∂2_u2_f,∂3_u2_f) = ∇(p.u2_fc,p)
	c1 = -im*scaprod(p.u1_fc,∂1_u2_f,p,true); c2 = -im*scaprod(p.u1_fc,∂2_u2_f,p,true)
	c1,c2
end

# the gauge is fixed such that <Φ1,(-i∇_x)Φ2> = <u1,(-i∇_x)u2> = vF[i;-1] where vF ∈ ℝ+, hence <e^(iξ) Φ1,(-i∇_x) e^(-iξ)Φ2> = e^(-2iξ) vF[i;-1]
# this is vF[i;-1] which has to be chosen, and not another one, for our 𝕍 to look like T_BM
function records_fermi_velocity_and_fixes_gauge(p) 
	(c1,c2) = compute_scaprod_Dirac(p)
	ratios = abs.([c1/c2+im,c1/c2-im]) # this can be one of the two, because of numerical instabilities choosing u1 and u2 which are interchangeable
	I = argmin(ratios)
	if I==2
		exchange_u1_u2(p)
		records_fermi_velocity_and_fixes_gauge(p) 
	else
		ratio = ratios[I]
		# px("Fermi velocity, distance to theory of the ratio:",ratio)
		@assert ratio<1e-4
		r,ξ = abs(c2),atan(imag(c2),real(c2))-π
		# px("<u1,(-i∇r) u2> = [i;",I==1 ? "-" : "+","1]*",r,"e^(i × ",ξ,") so vF=",r)
		p.v_fermi = r
		@assert imag(c2*cis(-ξ)) < 1e-10
		change_gauge_functions(ξ/2,p)

		# Verification that vF is real positive and close to vF
		(c1,c2) = compute_scaprod_Dirac(p)
		# px("new c1,c2= ",c1," ",c2)
		@assert norm([c1,c2] .- r*[im;-1])<1e-4
		px("<u1,(-i∇r) u2> = vF[i;-1] where vF=",r)
	end
end

function compute_vF_with_diagonalization_doesnt_work(p)
	(A1,A2) = (form_∇_one_matrix(p.u1_fb,p.u2_fb,1,p),form_∇_one_matrix(p.u1_fb,p.u2_fb,2,p))
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
	(α1,α2) = turn(p.u1_fb,p.u2_fb,U1)
	Aα1 = form_∇_one_matrix(α1,α2,1,p)
	px("Aα1")
	display(Aα1)

	(β1,β2) = turn(p.u1_fb,p.u2_fb,U2)
	Aβ2 = form_∇_one_matrix(β1,β2,2,p)
	px("Aβ2")
	display(Aβ2)

	px("Egalite: ",norm(U1.-U2))

	id = [1 0;0 1]
	px("Verifs id ",norm(W1'*W1 .- id)," ",norm(V1'*V1 .- id))
end

######################### Non local potential

function FormNonLocal(basis,T)
	model = basis.model

	# keep only pseudopotential atoms and positions
	psp_groups = [group for group in model.atom_groups
		      if model.atoms[first(group)] isa ElementPsp]
	psps          = [model.atoms[first(group)].psp      for group in psp_groups]
	psp_positions = [model.positions[group] for group in psp_groups]

	isempty(psp_groups) && return TermNoop()
	ops = map(basis.kpoints) do kpt
		P = DFTK.build_projection_vectors_(basis, kpt, psps, psp_positions)
		D = DFTK.build_projection_coefficients_(T, psps, psp_positions)
		DFTK.NonlocalOperator(basis, kpt, P, D)
	end
	DFTK.TermAtomicNonlocal(ops)
end

function extract_nonlocal(p)
	# \text{Energy} = \sum_a \sum_{ij} \sum_{n} f_n <ψ_n|p_{ai}> D_{ij} <p_{aj}|ψ_n>
	# Build projection vectors for a atoms array generated by term_nonlocal
	# H_at  = sum_ij Cij |pi> <pj|
	# H_per = sum_R sum_ij Cij |pi(x-R)> <pj(x-R)|
	# <e_kG'|H_per|e_kG> = 1/Ω sum_ij Cij pihat(k+G') pjhat(k+G)^*
	# where pihat(q) = ∫_R^3 pi(r) e^{-iqr} dr
	# We store 1/√Ω pihat(k+G) in proj_vectors.

	# psps = [p.psp,p.psp]; positions = [1,1] # POSITIONS ARE NOT USED IN DFTK
	# coefs = DFTK.build_projection_coefficients_(ComplexF64,psps,positions)
	# vecs = DFTK.build_projection_vectors_(p.basis,p.K_kpt,psps,positions)
	# Builds coefficients and projection
	px("Extracts non local potential")
	basis = PlaneWaveBasis(p.scfres.basis, [p.K_red_3d], [1])
	NLOp = FormNonLocal(basis,ComplexF64)
	# We store 1/√Ω ∫_Ω pi(r) e^{-i(K+G)r} dr in proj_vectors"
	# With our convention, we have fhat(k) ≃ (1/|Ω|) ∫_Ω pi(r) e^{-ikr} dr
	# so we need to divide by another 1/sqrt(|Ω|)
	vecs = NLOp.ops[1].P/sqrt(p.Vol)
	coefs = NLOp.ops[1].D # indices are i,j not a the indice of atoms
	px("Length projector ",size(vecs)," length coefs ",size(coefs)," values")
	display(coefs)
	p.non_local_coef = real(coefs[1,1])
	@assert imag(p.non_local_coef) < 1e-6
	@assert sum(abs.([coefs[1,1] 0;0 coefs[1,1]] .- coefs))/abs(coefs[1,1]) < 1e-8
	@assert sum(abs.(vecs[:,1].-conj.(vecs)[:,2]))/sum(abs.(vecs[:,1])) < 1e-8
	p.non_local_φ1_fb = vecs[:,1]
	p.non_local_φ2_fb = vecs[:,2]
	non_local_φ1_dir = G_to_r(p.basis,p.K_kpt,p.non_local_φ1_fb)
	non_local_φ2_dir = G_to_r(p.basis,p.K_kpt,p.non_local_φ2_fb)
	p.non_local_φ1_fc = myfft(non_local_φ1_dir,p.Vol)
	p.non_local_φ2_fc = myfft(non_local_φ2_dir,p.Vol)
end

function non_local_energy(u,p) # <u_nk, (V_nl)_k u_nk>, u is the periodic Bloch function, in the Fourier ball
	p.Vol_recip*
	sum(sum([abs2(u[g].*conj.(p.non_local_φ1_fb[g])
		      .*cis(p.shifts_atoms[a]⋅p.Gvectors_cart[g])) for g=1:length(u)]) for a=1:2)
end

function non_local_deriv_energy(n_samplings_over_2,p) # <u_nk, ∇_k (V_nl)_k u_nk>, has to be computed by finite diff
	@assert n_samplings_over_2 ≥ 4
	n_samplings = 2*n_samplings_over_2+1 # has to be even
	Dλ = 0.0001
	start_λ = 1-Dλ; end_λ = 1+Dλ; dλ = (end_λ-start_λ)/(n_samplings-1)
	set_coefs = [start_λ + i*dλ for i=0:n_samplings-1]
	set_cart_K = [λ*p.K_coords_cart for λ in set_coefs]
	values = []
	for i=1:n_samplings
		k = k_cart2red(set_cart_K[i],p)
		us = (diag_monolayer_at_k(k,p))[2]
		u = us[p.i_state]
		# px("len ",length(u))
		val = non_local_energy(u,p)
		push!(values,val)
	end
	Ipoint = Int(floor(n_samplings/4)-1)
	dK = norm(set_cart_K[Ipoint+1]-set_cart_K[Ipoint])
	x_axis = norm.(set_cart_K)
	pl = plot(x_axis,values)
	# savefig(pl,string(p.path_plots,"graph_non_loc_der.png"))
	nld = (values[Ipoint+1]-values[Ipoint])/dK
	px("Non local deriv: ",nld)
	nld
end

######################### Exports

function exports_v_u1_u2_φ(p)
	filename = string(p.path_exports,"N",p.N,"_Nz",p.Nz,"_u1_u2_V_nonLoc.jld")
	save(filename,"N",p.N,"Nz",p.Nz,"a",p.a,"L",p.L,"v_f",p.v_monolayer_fc,"u1_f",p.u1_fc,"u2_f",p.u2_fc,"v_fermi",p.v_fermi,"φ1_f",p.non_local_φ1_fc,"φ2_f",p.non_local_φ2_fc,"non_local_coef",p.non_local_coef,"shifts_atoms",p.shifts_atoms,"interlayer_distance",p.interlayer_distance)
	px("Exported : V, u1, u2 functions, Fermi velocity, and non local quantities, for N=",p.N,", Nz=",p.Nz)
end

function export_Vint(p)
	filename = string(p.path_exports,"N",p.N,"_Nz",p.Nz,"_Vint.jld")
	save(filename,"N",p.N,"Nz",p.Nz,"a",p.a,"L",p.L,"Vint_f",p.Vint_f)
	px("Exported : Vint for N=",p.N,", Nz=",p.Nz)
end

######################### Test symmetries

function test_rot_sym(p)
	# Tests
	(Ru0,Tu0) = (R(p.u0_fb,p),τ(p.u0_fb,p.shift_K,p)) # u0
	(RS,TS) = (R(p.u1_fb,p),τ(p.u1_fb,p.shift_K,p))   # u1
	(RW,TW) = (R(p.u2_fb,p),τ(p.u2_fb,p.shift_K,p))   # u2
	(Rφ1,Tφ1) = (R(p.non_local_φ1_fb,p),τ(p.non_local_φ1_fb,p.shift_K,p))
	(Rφ2,Tφ2) = (R(p.non_local_φ2_fb,p),τ(p.non_local_φ2_fb,p.shift_K,p))

	τau = cis(2π/3)
	# p.non_local_φ_fb = conj.(p.non_local_φ_fb)
	
	px("Test R Φ0 =     Φ0 ",norm(Ru0.-Tu0)/norm(Ru0))
	px("Test R Φ1 = τ   Φ1 ",norm(RS.-τau*TS)/norm(RS)) # R u1 = τ e^{ix⋅(1-R_{2π/3})K} u1 = τ e^{ix⋅[-1,0,0]a^*} u1
	px("Test R Φ2 = τ^2 Φ2 ",norm(RW.-τau^2*TW)/norm(RW))
	px("Test R φ1 = τ   φ1 ",norm(Rφ1.- τau*Tφ1)/norm(Rφ1))
	px("Test R φ2 = τ^2 φ2 ",norm(Rφ2.- τau^2*Tφ2)/norm(Rφ2))
end

######################### Plot functions

function plot_mean_V(p) # rapid plot of V averaged over XY, to see which are the right orders of magnitude
	Vz = [sum(p.v_monolayer_dir[:,:,z]) for z=1:p.Nz]/p.N^2
	pl = plot(Vz)
	savefig(pl,string(p.path_plots,"Vz.png"))
end

# array of Fourier coefs to cartesian direct function
function arr2fun(u_fc,p;bloch_trsf=true) # u in the Fourier ball, to function in cartesian
	K_cart = k_red2cart(p.K_red,p)
	plus_k = bloch_trsf ? K_cart : [0,0] # EQUIVALENT TO APPLY e^{iKx} !
	f(x,y,z) = 0
	for imx=1:p.N, imy=1:p.N, imz=1:p.Nz
		mx,my,mz = p.k_axis[imx],p.k_axis[imy],p.kz_axis[imz]
		if norm([mx,my,mz])<p.plots_cutoff
			ma = mx*p.a1_star + my*p.a2_star + plus_k
			g(x,y,z) = u_fc[imx,imy,imz]*cis([ma[1],ma[2],0]⋅[x,y,z])
			f = f+g
		end
	end
	f
end

function arr2fun_Gvectors(u,p;bloch_trsf=true) # u in the Fourier ball, to function in cartesian
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

function eval_f(g,res,p)
	# a = SharedArray{ComplexF64}(res,res,p.Nz)
	a = zeros(ComplexF64,res,res,p.Nz)
	indices = [(i,j,kiz) for i=0:res-1, j=0:res-1, kiz=1:p.Nz]
	aa = p.a
	kt = p.kz_axis
	# for l=1:length(indices) # 9.5s
	# @time @distributed for l=1:length(indices) # 
	Threads.@threads for l=1:length(indices) # 5s
		(i,j,kiz) = indices[l]
		# px("truc ",a[i,j,kiz])
		a[i+1,j+1,kiz] = g(i*aa/res,j*aa/res,kt[kiz])
	end
	a
end

# eval_f(g,res,p) = [g(i*p.a/res,j*p.a/res,p.kz_axis[kiz]) for i=0:res-1, j=0:res-1, kiz=1:p.Nz]

function simple_plot(u,fun,Z,p;n_motifs=3,bloch_trsf=true,res=25)
	f = arr2fun(u,p;bloch_trsf=bloch_trsf)
	g = scale_fun3d(f,n_motifs)
	a = fun.(eval_f(g,res,p))
	b = intZ(abs.(a),p)
	# b = a[:,:,Z]
	heatmap(b,size=(1000,1000))
end

function rapid_plot(u,p;n_motifs=5,name="rapidplot",bloch_trsf=true,res=25)
	Z = 10
	funs = [abs,real,imag]
	hm = [simple_plot(u,fun,Z,p;n_motifs=n_motifs,bloch_trsf=bloch_trsf,res=res) for fun in funs]
	plot_z = plot(intXY(real.(ifft(u,p.L)),p))
	push!(hm,plot_z)
	size = 600
	r = length(hm)
	pl = plot(hm...,layout=(r,1),size=(size+100,r*size),legend=false)
	# pl = plot(hm...,layout=(1,r),size=(r*size,size-200),legend=false)
	full_name = string(p.path_plots,name,"_N",p.N,"_Nz",p.Nz)
	savefig(pl,string(full_name,".png"))

	# Fourier
	uXY = intZ(abs.(u),p)
	uZ = intXY(abs.(u),p)
	plXY = heatmap(uXY,size=(1500,1000))
	plZ = plot(uZ,size=(1500,1000))
	pl = plot(plXY,plZ)
	savefig(pl,string(full_name,"_fourier.png"))
	px("Plot of ",name," done")
end

function plot_Vint(p)
	v_bilayer_f = [p.V_bilayer_Xs_fc[1,1,miz] for miz=1:p.Nz]/sqrt(p.cell_area)
	# average of v_monolayer
	v_f = myfft([sum(p.v_monolayer_dir[:,:,z]) for z=1:p.Nz]/p.N^2,p.L)
	# shifts of ±d/2 of the monolayer KS potential
	v_f_plus  = v_f.*cis.( 2π*p.kz_axis*p.interlayer_distance/(2*p.L))
	v_f_minus = v_f.*cis.(-2π*p.kz_axis*p.interlayer_distance/(2*p.L))

	# Builds u1 and u2
	abs2_u1_averaged = p.cell_area*[sum(abs2.(p.u1_dir)[:,:,z]) for z=1:p.Nz]/p.N^2
	abs2_u1_f = myfft(abs2_u1_averaged,p.L)
	abs2_u1_f_plus = abs2_u1_f.*cis.( 2π*p.kz_axis*p.interlayer_distance/(2*p.L))
	abs2_u1_f_minus = abs2_u1_f.*cis.(-2π*p.kz_axis*p.interlayer_distance/(2*p.L))

	res = 700
	v_int = eval_fun_to_plot_1d(p.Vint_f,res,p.L)
	v_plus = eval_fun_to_plot_1d(v_f_plus,res,p.L)
	v_minus = eval_fun_to_plot_1d(v_f_minus,res,p.L)
	v_bilayer = eval_fun_to_plot_1d(v_bilayer_f,res,p.L)
	abs2_u1_plus = eval_fun_to_plot_1d(abs2_u1_f_plus,res,p.L)
	abs2_u1_minus = eval_fun_to_plot_1d(abs2_u1_f_minus,res,p.L)

	z_axis = (0:res-1)*p.L/res .- p.L/2

	labels = [L"V_{\rm int, d}(z)" L"\frac{1}{|\Omega|} \int_\Omega V(x,z+d/2) d x" L"\frac{1}{|\Omega|} \int_\Omega V(x,z-d/2) d x" L"\frac{1}{|\Omega|^2} \int_{\Omega \times \Omega} V^{(2)}_{d,\bf{y}}(\bf{x},z) d \bf{x} d \bf{y}" L"\int_\Omega |\Phi_1|^2(x,z+d/2) d x" L"\int_\Omega |\Phi_1|^2(x,z-d/2) d x"]
	pl = plot(z_axis,fftshift.([v_int,v_plus,v_minus,v_bilayer,abs2_u1_plus,abs2_u1_minus]),label=labels,size=(900,400),legend = :outerright, legendfontsize=10)
	savefig(pl,string(p.path_plots,"Vint.png"))

	if p.export_plots_article
		savefig(pl,string(p.path_plots_article,"Vint.pdf"))
	end
end

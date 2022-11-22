include("common_functions.jl")
using DFTK, LinearAlgebra, JLD, LaTeXStrings
using Plots, CairoMakie, DataFrames
setup_threading()
px = println

######################### Parameters

# Class containing all the parameters and functions which need to be stored
mutable struct Params
    # Lattice
    dim # dimension
    a; a1; a2; a1_star; a2_star
    lattice_2d
    lattice_3d
    graphene_lattice_orientation # type 2π/3 (true) if there is a rotation of 2π/3 between a1 and a2, type π/3 otherwise

    # Discretization numbers
    N; Nz; N2d; N3d
    n_fball # size of the Fermi ball ∈ ℕ

    # Infinitesimal quantities
    dx; dS; dz; dv

    # Axis
    x_axis_cart
    k_axis; k_grid
    kz_axis
    x_axis_red; z_axis_red; z_axis_cart

    # Volumes, surfaces, distances
    sqrtVol
    sqi # 1/sqrt(cell_area)
    L # physical length of periodicity for the computations for a monolayer
    cell_area
    Vol # Volume Ω×L of the direct lattice
    Vol_recip # Volume of the reciprocal lattice

    # Matrices
    M3d # M = (a1,a2), M3d = (a1,a2,L*e3) is the lattice
    R_four_2d # rotation of 2π/3, in Fourier labels
    M_four_2d # mirror

    # Dirac point quantities
    K_red
    K_red_3d; K_coords_cart; K_kpt # Dirac point in several formats
    shift_K_R; shift_K_M; shift_K
    v_fermi # Fermi velocity
    gauge_param # ∈ ±1, so that <Φ1,(-i∇_x)Φ2> = gauge_param*vF*[1,-i]

    ### Monolayer functions
    i_state # index of first valence state
    C1; C2 # shift, in reduced coordinates, of the carbon atoms
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
    plots_cutoff # Fourier cutoff for the plane-wave basis in plots
    root_path; path_exports; path_plots # paths
    export_plots_article
    path_plots_article
    alleviate # alleviate computations, to work on plots for instance, doing computations quickly
    function Params()
        p = new()
        p.export_plots_article = false
        p
    end
end

function init_params(p)
    init_cell_vectors(p;moire=false)
    p.Vol = DFTK.compute_unit_cell_volume(p.lattice_3d)
    p.Vol_recip = (2π)^3/p.Vol
    p.recip_lattice = DFTK.compute_recip_lattice(p.lattice_3d)
    p.recip_lattice_inv = inv(p.recip_lattice)

    # Dirac point
    p.K_red_3d = vcat(p.K_red,[0])
    p.K_coords_cart = k_red2cart_3d(p.K_red_3d,p)
    # (1-R_{2π/3})K = m a^*, shift of this vector
    p.shift_K_R = [myfloat2int.((I-matrix_rot_red(2π/3,p))*p.K_red;name="shiftKR");0]
    p.shift_K_M = [myfloat2int.((I-           p.M_four_2d)*p.K_red;name="shiftKM");0]
    # p.shift_K =   [myfloat2int.(p.K_red;name="shiftK");0]

    # Paths
    p.root_path = "apply_graphene_outputs/"
    p.path_exports = p.root_path*"exported_functions/"
    p.path_plots = p.root_path*"plots/"
    p.path_plots_article = "plots_article/"
    create_dir(p.root_path)
    create_dir(p.path_exports)
    create_dir(p.path_plots)
    create_dir(p.path_plots_article)
    p
end


######################### Operations on functions

# from a list G=[a,b,c] of int, gives the iG such that Gvectors[iG]==G. If G is not in it, gives nothing
function index_of_Gvector(G,p) 
    tupleG = Tuple(G)
    if haskey(p.Gvectors_inv,tupleG)
        return p.Gvectors_inv[tupleG]
    else
        return nothing
    end
end

# Generates the operator doing U_m -> V_m := U_{Lm}
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

# (Ru)_G = u_{M G} or (Ru)^D_m = u^D_{F^{-1} M F(m)}, where ^D means that we take the discretization vector
function R_fb(u,p)
    L(G) = M2d_2_M3d(p.R_four_2d)*G
    OpL(u,p,L)
end

function G_fb(u,p) # mirror
    L(h) = M2d_2_M3d(p.M_four_2d)*h
    OpL(u,p,L)
end

# τ is equivalent to multiplication by e^{i cart(k) x), where k is in reduced, e^{i m0 a^*⋅x} u = ∑ e^{ima^*⋅x} u_{m-m0}
τ(u,k,p) = OpL(u,p,G -> G .- k)
P_fb(u,p) = OpL(u,p,G -> -G) # parity

z_translation(a,Z,p) = [a[x,y,mod1(z-Z,p.Nz)] for x=1:p.N, y=1:p.N, z=1:p.Nz] # translation on Z
r_translation(a,s,p) = [a[mod1(x-s[1],p.N),mod1(y-s[2],p.N),z] for x=1:p.N, y=1:p.N, z=1:p.Nz] # translation on XY, s ∈ {0,…,p.N-1}^2, 0 for no translation

######################### Solves SCF or Schrödinger

# Call SCF to get the Kohn-Sham potential of the monolayer
function scf_graphene_monolayer(p)
    # Loads model
    p.psp = load_psp("hgh/pbe/c-q4")
    # Choose carbon atoms
    C = ElementPsp(:C, psp=p.psp)
    # Puts carbon atoms at the right positions for graphene
    minus = p.graphene_lattice_orientation ? -1 : 1
    p.C1 = [1/3,minus*1/3,0.0]; p.C2 = -p.C1
    p.shifts_atoms = [p.C1,p.C2]
    # Builds model
    model = model_PBE(p.lattice_3d, [C,C], p.shifts_atoms; temperature=1e-3, smearing=Smearing.Gaussian())
    # Builds basis
    basis = PlaneWaveBasis(model; Ecut=p.ecut, p.kgrid)
    # px("Number of spin components ",model.n_spin_components)
    # Loads sizes
    (p.N,p.N,p.Nz) = basis.fft_size
    # Generates the right path strings for plots
    p.path_plots = string(p.root_path,"plots_N",p.N,"_Nz",p.Nz,"/")
    create_dir(p.path_plots)
    # Builds axis
    p.N3d = p.N^2*p.Nz
    p.x_axis_red = ((0:p.N -1))/p.N
    p.z_axis_red = ((0:p.Nz-1))/p.Nz
    p.z_axis_cart = p.z_axis_red*p.L
    # Do initializations
    init_cell_infinitesimals(p;moire=false)
    if !p.alleviate
        @assert abs(p.dv - basis.dvol) < 1e-10
    end
    # Extracts kpoint information
    i_kpt = 1
    kpt = basis.kpoints[i_kpt]
    # Runs SCF
    px("Runs DFTK's SCF algorithm for the monolayer...")
    p.scfres = self_consistent_field(basis)
    px("Energies")
    display(p.scfres.energies)
    px("DFTK's SCF algorithm ended")
    # Extracts the Kohn-Sham potential
    p.v_monolayer_dir = DFTK.total_local_potential(p.scfres.ham)[:,:,:,1]
    substract_by_far_value(p.v_monolayer_dir,p)
    p.v_monolayer_fc = myfft(p.v_monolayer_dir,p.Vol)

    occupation = floor(Int,sum(p.scfres.ρ)*p.dv +0.5)
    px("Number of effective particles in the model: ",occupation)
    # Extracts Kohn-Sham orbitals
    for i=1:occupation
        ui = p.scfres.ψ[i_kpt][:, i]
        ui_dir = G_to_r(basis,kpt,ui)
    end
end

# Compute the band structure at momentum k, when the KS potential is already known
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
    # Puts carbon atoms at the right places
    stacking_shift_red = [sx,sy,0.0]
    D = [0;0;p.interlayer_distance/(p.L*2)]

    c1_plus =  p.C1 .+ D .+ stacking_shift_red
    c2_plus =  p.C2 .+ D .+ stacking_shift_red
    c1_moins = p.C1 .- D
    c2_moins = p.C2 .- D
    C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
    positions = [c1_plus,c2_plus,c1_moins,c2_moins]
    n_extra_states = 1

    if D==0 px("CANNOT DO DFT WITH d=0") end
    # Builds model
    model = model_PBE(p.lattice_3d, [C,C,C,C], positions; temperature=1e-3, smearing=Smearing.Gaussian())#, symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=p.ecut, kgrid=p.kgrid)
    if !p.alleviate
        b = (p.N,p.N,p.Nz) == basis.fft_size
        if !b
            px("Problem in scf bilayer, (N,Nz)=(",p.N,",",p.Nz,") but fftsize=",basis.fft_size,". To solve the issue, change ecut !")
            @assert false
        end
        @assert abs(p.dv - basis.dvol) < 1e-10
        @assert p.x_axis_red == first.(DFTK.r_vectors(basis))[:,1,1]
    end
    # Does SCF
    scfres = self_consistent_field(basis;tol=p.tol_scf,n_ep_extra=n_extra_states,eigensolver=lobpcg_hyper,maxiter=100,callback=x->nothing)
    # Extracts the Kohn-Sham potential
    Vks_dir = DFTK.total_local_potential(scfres.ham)[:,:,:,1] # select first spin component
    substract_by_far_value(Vks_dir,p)
    myfft(Vks_dir,p.Vol)
end

# Obtains u0, u1 and u2 at the Dirac point
function get_dirac_eigenmodes(p)
    # Do the diagonalization
    (Es_K,us_K,basis,ham) = diag_monolayer_at_k(p.K_red_3d,p)
    # Extracts basis
    p.basis = basis
    p.K_kpt = basis.kpoints[1]
    K_norm_cart = 4π/(3*p.a)
    @assert abs(K_norm_cart - norm(k_red2cart_3d(p.K_kpt.coordinate,p))) < 1e-10

    # Extracts dual vectors
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
    p.u1_fb = us_K[p.i_state]
    p.u2_fb = us_K[p.i_state+1]
    update_u1_u2(p)

    # Verifications. Don't do them when the size of matrices is large, this will crash
    # H_K_dirac = ham.blocks[1]; Hmat = Array(H_K_dirac)
    # res = norm(H_K_dirac * p.u1_fb - dot(p.u1_fb, H_K_dirac * p.u1_fb) * p.u1_fb)
    # println("Verification residual norm ",res," eigenvalues (p.u1_fb,p.u2_fb) ",real(dot(p.u1_fb, H_K_dirac * p.u1_fb)),",",real(dot(p.u2_fb, H_K_dirac * p.u2_fb))," shoul equal ",real(Es_K[p.i_state]),",",real(Es_K[p.i_state+1])," Norm p.u1_fb ",norm(p.u1_fb))
    nothing
end

# Creates u0 u1 and u2 in direct and Fourier cube representations
function update_u1_u2(p)
    # Computes them in direct space
    p.u0_dir = G_to_r(p.basis,p.K_kpt,p.u0_fb)
    p.u1_dir = G_to_r(p.basis,p.K_kpt,p.u1_fb)
    p.u2_dir = G_to_r(p.basis,p.K_kpt,p.u2_fb)
    # Computes them in the Fourier cube
    p.u0_fc = myfft(p.u0_dir,p.Vol)
    p.u1_fc = myfft(p.u1_dir,p.Vol)
    p.u2_fc = myfft(p.u2_dir,p.Vol)
end

######################### Coordinates changes

# coords in reduced to coords in cartesian
k_red2cart_3d(k_red,p) = p.recip_lattice*k_red
# coords in cartesian to coords in reduced
k_cart2red(k_cart,p) = p.recip_lattice_inv*k_cart

######################### Obtain the good orthonormal basis for the periodic Bloch functions at Dirac points u1 and u2

# Turns the degenerate eigenvectors u1 and u2 in SU(2) so that the physical associated eigenvectors respect the symmetries R ϕj = ω^j ϕj. See documentation to see the details on this operation
function get_natural_basis_u1_and_u2(p)
    τau = cis(2π/3)
    (Ru1,Tu1) = (R_fb(p.u1_fb,p),τ(p.u1_fb,p.shift_K_R,p))
    (Ru2,Tu2) = (R_fb(p.u2_fb,p),τ(p.u2_fb,p.shift_K_R,p))
    d1 = Ru1.-τau*Tu1
    d2 = Ru2.-τau*Tu2

    c = (norm(d1))^2
    s = d1'*d2
    f = (c/abs(s))^2

    U = (s/abs(s))/(sqrt(1+f))*p.u1_fb + (1/sqrt(1+1/f))*p.u2_fb
    V = (s/abs(s))/(sqrt(1+f))*p.u1_fb - (1/sqrt(1+1/f))*p.u2_fb

    (RU,TU) = (R_fb(U,p),τ(U,p.shift_K_R,p))
    (RV,TV) = (R_fb(V,p),τ(V,p.shift_K_R,p))

    I = argmin([norm(RU.-τau *TU),norm(RV.-τau *TV)])
    p.u1_fb = I == 1 ? U : V

    # ϕ2(x) = conj(ϕ1(-x)) ⟹ u2(x) = conj(u1(-x))
    # conj ∘ parity is conj in Fourier
    p.u2_fb = conj.(p.u1_fb)
    update_u1_u2(p)
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

function form_∇2_term(u,w,l,j,p) # <(-i∂_ℓ +K_ℓ)u,(-i∂_j +K_j)w>
    GpKl = [p.Gplusk_vectors_cart[iG][l] for iG=1:p.n_fball]
    GpKj = [p.Gplusk_vectors_cart[iG][j] for iG=1:p.n_fball]
    sum((conj.(GpKl.*u)) .* (GpKj.*w))
end

# 2×2 matrices <u,(-i∂_j +K_j)u>
form_∇_one_matrix(u1,u2,j,p) = Hermitian([form_∇_term(u1,u1,j,p) form_∇_term(u1,u2,j,p);form_∇_term(u2,u1,j,p) form_∇_term(u2,u2,j,p)])

function fermi_velocity_from_scalar_products(p) # Computes the Fermi velocity from the formula <ϕ_p,(-i∇)ϕ_j> = vF σ. Needs that u1 and u2 are rotated before calling
    (A1,A2) = (form_∇_one_matrix(p.u1_fb,p.u2_fb,1,p),form_∇_one_matrix(p.u1_fb,p.u2_fb,2,p))
    # display(A1); display(A2)
    px("Fermi velocity from rotated u1 and u2 ",abs(A1[1,2]))
end

function fermi_velocity_from_scalar_products(p) # <ϕ_i,(-i∇) P ϕ_j> scalar products with parity operation
    u1 = p.u1_fb; u2 = p.u2_fb
    Pu1 = P_fb(u1,p); Pu2 = P_fb(u2,p)
    v12 = form_∇_term(u1,u2,1,p); v12_ = form_∇_term(u1,u2,2,p)
    vP12 = form_∇_term(u1,Pu2,1,p); vP12_ = form_∇_term(u1,Pu2,2,p)
    vP21 = form_∇_term(u2,Pu1,1,p); vP21_ = form_∇_term(u2,Pu1,2,p)
    c = hartree_to_ev
    px("Fermi velocity <ϕ1,(-i∇) ϕ2> =(",c*v12,",",c*v12_,") eV \n <ϕ1,(-i∇) P ϕ2>",c*vP12," ",c*vP12_," eV, \n<ϕ2,(-i∇) P ϕ1> ",c*vP21," ",c*vP21_," eV")
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
    pl = Plots.plot(x_axis,[values_down,values_up])
    savefig(pl,p.path_plots*"graph_fermi_velocity.png")
    vF = (values_down[Ipoint+1]-values_down[Ipoint])/dK
    vF_up = (values_up[Ipoint+1]-values_up[Ipoint])/dK
    px("Fermi velocity: ",vF,". Verification with upper eigenvalue: ",vF_up)
    vF
end

double_der_matrix(u,w,p) = [form_∇2_term(u,w,i,j,p) for i=1:2, j=1:2]

function double_derivatives_scalar_products(p)
    u1 = p.u1_fb; u2 = p.u2_fb
    M11 = double_der_matrix(u1,u1,p)
    M12 = double_der_matrix(u1,u2,p)
    c = hartree_to_ev
    px("Double derivatives scalar products in eV\n","<∂i ϕ1,∂j ϕ1> :")
    display(c*M11)
    px("<∂i ϕ1,∂j ϕ2> :")
    display(c*M12)
    px("<∂z ϕ1,∂z ϕ1> : ",c*form_∇2_term(u1,u1,3,3,p))
    px("<∂z ϕ1,∂z ϕ2> : ",c*form_∇2_term(u1,u2,3,3,p))
end

function change_gauge_functions(ξ,p) # changes the phasis of functions
    α = cis(ξ); β = cis(-ξ)
    p.u0_fb *= α; p.u1_fb *= α; p.u2_fb *= β
    p.u0_fc *= α; p.u1_fc *= α; p.u2_fc *= β
    p.u0_dir *= α; p.u1_dir *= α; p.u2_dir *= β
    p.non_local_φ1_fb *= α; p.non_local_φ1_fc *= α
    p.non_local_φ2_fb *= β; p.non_local_φ2_fc *= β
end

function compute_scaprod_∇(u,v,p) # <u,(-i∇_x)v>
    (∂1_v,∂2_v,∂3_v) = ∇(v,p)
    c1 = -im*scaprod(u,∂1_v,p,true); c2 = -im*scaprod(u,∂2_v,p,true)
    c1,c2
end

compute_scaprod_Dirac_u1_u2(p) = compute_scaprod_∇(p.u1_fc,p.u2_fc,p)

function compute_scaprod_Dirac_u1_u1(p)
    c1,c2 = compute_scaprod_∇(p.u1_fc,p.u1_fc,p)
    A = k_red2cart(p.K_red,p)
    (c1+A[1],c2+A[2])
end

# Changes the gauge such that <Φ1,(-i∇_x)Φ2> = vF*[1,-i], vF ∈ ℝ+
# the gauge is fixed such that <Φ1,(-i∇_x)Φ2> = <u1,(-i∇_x)u2> = vF*goal where vF ∈ ℝ+, hence <e^(iξ) Φ1,(-i∇_x) e^(-iξ)Φ2> = e^(-2iξ) vF*goal
# this is vF*goal which has to be chosen, and not another one, for our 𝕍 to look like T_BM
function records_fermi_velocity_and_fixes_gauge(p) 
    goal = p.gauge_param*[1;-im]
    (c1,c2) = compute_scaprod_Dirac_u1_u2(p)
    # px("Values of Fermi velocity params before the gauge change ",c1," ",c2)
    if !p.alleviate
        @assert abs(goal[1]/goal[2] - c1/c2) <1e-4
    end
    I_real = imag(goal[1])<1e-8 ? 1 : 2
    fact = real(goal[I_real])*(I_real==1 ? c1 : c2)
    r,ξ = polar(fact)
    p.v_fermi = r
    @assert imag(fact*cis(-ξ)) < 1e-10
    change_gauge_functions(ξ/2,p)

    # Verification that vF is real positive and close to vF
    (c1,c2) = compute_scaprod_Dirac_u1_u2(p)
    if !p.alleviate
        @assert norm([c1,c2] .- r*goal)<1e-4
    end
    px("<u1,(-i∇r) u2> = vF[",goal[1],",",goal[2],"] where vF=",r)
end

######################### Non local potential

function FormNonLocal(basis,T) # taken from DFTK, to extract non-local potential quantities
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
    # From DFTK:
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
    basis = PlaneWaveBasis(p.scfres.basis, [p.K_red_3d], [1])
    NLOp = FormNonLocal(basis,ComplexF64)
    # We store 1/√Ω ∫_Ω pi(r) e^{-i(K+G)r} dr in proj_vectors"
    # With our convention, we have fhat(k) ≃ (1/|Ω|) ∫_Ω pi(r) e^{-ikr} dr
    # so we need to divide by another 1/sqrt(|Ω|)
    vecs = NLOp.ops[1].P
    coefs = NLOp.ops[1].D # indices are i,j not a the indice of atoms
    # px("Length projector ",size(vecs)," length coefs ",size(coefs)," values")
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
    # px("Non local coef ",p.non_local_coef)
end

function non_local_energy(u,p) # <u_nk, (V_nl)_k u_nk>, u is the periodic Bloch function, in the Fourier ball
    # p.Vol_recip*
    # sum(sum([abs2(u[g].*conj.(p.non_local_φ1_fb[g])
                  # .*cis(p.shifts_atoms[a]⋅p.Gvectors_cart[g])) for g=1:length(u)]) for a=1:2)
    p.non_local_coef*(abs2(u'*p.non_local_φ1_fb) + abs2(u'*p.non_local_φ2_fb))
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
    pl = Plots.plot(x_axis,values)
    # savefig(pl,string(p.path_plots,"graph_non_loc_der.png"))
    nld = (values[Ipoint+1]-values[Ipoint])/dK
    px("Non local deriv: ",nld,", to be compared to the order of magnitude of potential energy ",scaprod(p.v_monolayer_fc,abs2.(p.u1_fc),p,true))
    nld
end

######################### Exports

function exports_v_u1_u2_φ(p)
    marker_gauge = p.gauge_param == 1 ? "p" : "m"
    filename = string(p.path_exports,"N",p.N,"_Nz",p.Nz,"_g",marker_gauge,"_u1_u2_V_nonLoc.jld")
    save(filename,"N",p.N,"Nz",p.Nz,"a",p.a,"L",p.L,"v_f",p.v_monolayer_fc,"u1_f",p.u1_fc,"u2_f",p.u2_fc,"v_fermi",p.v_fermi,"φ1_f",p.non_local_φ1_fc,"φ2_f",p.non_local_φ2_fc,"non_local_coef",p.non_local_coef,"shifts_atoms",p.shifts_atoms,"gauge_param",p.gauge_param)
    px("Exported : V, u1, u2 functions, Fermi velocity, and non local quantities, for N=",p.N,", Nz=",p.Nz,", gauge_param=",p.gauge_param," at ",filename)
end

function export_Vint(p)
    marker_gauge = p.gauge_param == 1 ? "p" : "m"
    filename = string(p.path_exports,"N",p.N,"_Nz",p.Nz,"_g",marker_gauge,"_d",p.interlayer_distance,"_Vint.jld")
    save(filename,"N",p.N,"Nz",p.Nz,"a",p.a,"L",p.L,"d",p.interlayer_distance,"Vint_f",p.Vint_f)
    px("Exported : Vint for N=",p.N,", Nz=",p.Nz)
end

######################### Test symmetries

function test_normalizations(p)
    # Tests normalization
    # px("Normalization of u1: ",norms(p.u1_dir,p,false))
    # px("Orthonormality |<u1,u2>|= ",abs(scaprod(p.u1_fc,p.u2_fc,p)) + abs(scaprod(p.u1_dir,p.u2_dir,p,false)))
    @assert abs(norms(p.u1_dir,p,false)-1) + abs(scaprod(p.u1_fc,p.u2_fc,p)) + abs(scaprod(p.u1_dir,p.u2_dir,p,false)) < 1e-5
end

function test_rot_sym(p)
    # Tests
    (Ru0,Tu0) = (R_fb(p.u0_fb,p),τ(p.u0_fb,p.shift_K_R,p)) # u0
    (RS,TS) = (R_fb(p.u1_fb,p),τ(p.u1_fb,p.shift_K_R,p))   # u1
    (RW,TW) = (R_fb(p.u2_fb,p),τ(p.u2_fb,p.shift_K_R,p))   # u2
    (Rφ1,Tφ1) = (R_fb(p.non_local_φ1_fb,p),τ(p.non_local_φ1_fb,p.shift_K_R,p))
    (Rφ2,Tφ2) = (R_fb(p.non_local_φ2_fb,p),τ(p.non_local_φ2_fb,p.shift_K_R,p))

    τau = cis(2π/3)
    # p.non_local_φ_fb = conj.(p.non_local_φ_fb)

    px("Test R Φ0 =     Φ0 ",distance(Ru0,Tu0))
    px("Test R Φ1 = τ   Φ1 ",distance(RS,τau*TS)) # R u1 = τ e^{ix⋅(1-R_{2π/3})K} u1 = τ e^{ix⋅[-1,0,0]a^*} u1
    px("Test R Φ2 = τ^2 Φ2 ",distance(RW,τau^2*TW))
    px("Test R φ1 = τ   φ1 ",distance(Rφ1, τau*Tφ1))
    px("Test R φ2 = τ^2 φ2 ",distance(Rφ2, τau^2*Tφ2))
    px("Test R v  =      v ",distance(p.v_monolayer_fc,R_four(p.v_monolayer_fc,p)))
end

function test_mirror_sym(p)
    px("Tests for mirror symmetry, Gf(x1,x2,z) := f(x1,-x2,z)")
    (Gu0,Tu0) = (G_fb(p.u0_fb,p),τ(p.u0_fb,p.shift_K_M,p))
    (Gu1,Tu1) = (G_fb(p.u1_fb,p),τ(p.u1_fb,p.shift_K_M,p))
    (Gu2,Tu2) = (G_fb(p.u2_fb,p),τ(p.u2_fb,p.shift_K_M,p))

    Gu0 = G_fb(p.u0_fb,p)
    Gu1 = G_fb(p.u1_fb,p)
    Gu2 = G_fb(p.u2_fb,p)

    px("Test Φ1(-z)=-Φ1(z) ",distance(parity_z(p.u1_fc,p),-p.u1_fc))
    px("Test Φ2(-z)=-Φ2(z) ",distance(parity_z(p.u2_fc,p),-p.u2_fc))

    px("Test G Φ0 =  Φ0 ",distance(Gu0,Tu0))
    px("Test G Φ1 = ",p.gauge_param==1 ? "-" : " ","Φ2 ",distance(Gu1,-p.gauge_param*Tu2))
    px("Test G v  =   v ",distance(p.v_monolayer_fc,M_four(p.v_monolayer_fc,p)))
end

######################### Plot functions

function plot_mean_V(p) # rapid plot of V averaged over XY
    Vz = real.([sum(p.v_monolayer_dir[:,:,z]) for z=1:p.Nz])/p.N^2
    pl = Plots.plot(Vz)
    savefig(pl,p.path_plots*"V_averaged_over_xy.png")
end

# array of Fourier coefs to cartesian direct function
function arr2fun(u_fc,p;bloch_trsf=true,k_p=-1) # u in the Fourier ball, to function in cartesian
    K_cart = k_red2cart(p.K_red,p)
    plus_k = bloch_trsf ? K_cart : [0,0] # EQUIVALENT TO APPLY e^{iKx} !
    f(x,y,z) = 0
    for imx=1:p.N, imy=1:p.N, imz=1:p.Nz
        mx,my,mz = p.k_axis[imx],p.k_axis[imy],p.kz_axis[imz]
        if norm([mx,my,mz])<p.plots_cutoff
            ma = mx*p.a1_star + my*p.a2_star + (k_p==-1 ? plus_k : k_p)
            g(x,y,z) = u_fc[imx,imy,imz]*cis([ma[1],ma[2],0]⋅[x,y,z])
            f = f+g
        end
    end
    f
end

function eval_f(g,res,p) # evaluates the function g
    # a = SharedArray{ComplexF64}(res,res,p.Nz)
    a = zeros(ComplexF64,res,res,p.Nz)
    indices = [(i,j,kiz) for i=0:res-1, j=0:res-1, kiz=1:p.Nz]
    aa = p.a
    kt = p.kz_axis
    # for l=1:length(indices) # 9.5s
    Threads.@threads for l=1:length(indices) # 5s
        (i,j,kiz) = indices[l]
        a[i+1,j+1,kiz] = g(i*aa/res,j*aa/res,kt[kiz])
    end
    a
end

eval_f(g,res,p) = [g(i*p.a/res,j*p.a/res,p.kz_axis[kiz]) for i=0:res-1, j=0:res-1, kiz=1:p.Nz]
scale_fun3d(f,λ) = (x,y,z) -> f(λ*x,λ*y,λ*z)

function simple_plot(u,fun,Z,p;n_motifs=3,bloch_trsf=true,res=25,k_p=-1)
    f = arr2fun(u,p;bloch_trsf=bloch_trsf,k_p=k_p)
    g = scale_fun3d(f,n_motifs)
    a = fun.(eval_f(g,res,p))
    b = intZ(a,p)
    # b = a[:,:,Z]
    Plots.heatmap(b,size=(1000,1000), aspect_ratio=:equal)
end

# plots a function in both direct and Fourier space, with absolute value, real and imaginary parts, at some Z coordinate
function rapid_plot(u,p;n_motifs=5,name="rapidplot",bloch_trsf=true,k_p=-1,res=25)
    Z = 5
    funs = [abs,real,imag]
    hm = [simple_plot(u,fun,Z,p;n_motifs=n_motifs,bloch_trsf=bloch_trsf,res=res,k_p=k_p) for fun in funs]
    plot_z = Plots.plot(intXY(real.(myifft(u,p.L)),p))
    push!(hm,plot_z)
    size = 600
    r = length(hm)
    pl = Plots.plot(hm...,layout=(r,1),size=(size+100,r*size),legend=false)
    # pl = Plots.plot(hm...,layout=(1,r),size=(r*size,size-200),legend=false)
    full_name = string(p.path_plots,name,"_N",p.N,"_Nz",p.Nz)
    savefig(pl,full_name*".png")

    # Fourier
    uXY = intZ(abs.(u),p)
    uZ = intXY(abs.(u),p)
    plXY = Plots.heatmap(uXY,size=(1500,1000))
    plZ = Plots.plot(uZ,size=(1500,1000))
    pl = Plots.plot(plXY,plZ)
    savefig(pl,full_name*"_fourier.png")
    px("Plot of ",name," done")
end

######################### Plot Vint for the article

function red_arr2fun_red_1d(ψ_four,vol) # 1d, Fourier coefs to direct function
    N = length(ψ_four)
    k_axis = fftfreq(N)*N
    f(x) = 0
    for i=1:N
        g(x) = ψ_four[i]*cis(2π*k_axis[i]*x)/sqrt(vol)
        f = f + g
    end
    f
end

function eval_fun_to_plot_1d(ψ_four,res,vol) # 1d, Fourier coefs to direct on a grid
    f = red_arr2fun_red_1d(ψ_four,vol)
    real.(f.((0:res-1)/res))
end

function plot_Vint(p)
    v_bilayer_f = [p.V_bilayer_Xs_fc[1,1,miz] for miz=1:p.Nz]/sqrt(p.cell_area)
    # average of v_monolayer
    v_f = myfft(average_over_xy(p.v_monolayer_dir,p),p.L)
    # shifts of ±d/2 of the monolayer KS potential
    v_f_plus  = v_f.*cis.( 2π*p.kz_axis*p.interlayer_distance/(2*p.L))
    v_f_minus = v_f.*cis.(-2π*p.kz_axis*p.interlayer_distance/(2*p.L))

    # Builds u1 and u2
    abs2_u1_averaged = p.cell_area*average_over_xy(abs2.(p.u1_dir),p)
    abs2_u1_f = myfft(abs2_u1_averaged,p.L)
    abs2_u1_f_plus = abs2_u1_f.*cis.( 2π*p.kz_axis*p.interlayer_distance/(2*p.L))
    abs2_u1_f_minus = abs2_u1_f.*cis.(-2π*p.kz_axis*p.interlayer_distance/(2*p.L))

    res = 1000
    v_int = hartree_to_ev*eval_fun_to_plot_1d(p.Vint_f,res,p.L)
    v_plus = hartree_to_ev*eval_fun_to_plot_1d(v_f_plus,res,p.L)
    v_minus = hartree_to_ev*eval_fun_to_plot_1d(v_f_minus,res,p.L)
    v_bilayer = hartree_to_ev*eval_fun_to_plot_1d(v_bilayer_f,res,p.L)
    abs2_u1_plus = eval_fun_to_plot_1d(abs2_u1_f_plus,res,p.L)
    abs2_u1_minus = eval_fun_to_plot_1d(abs2_u1_f_minus,res,p.L)


    # pl = Plots.plot(z_axis,fftshift.([v_int,v_plus,v_minus,v_bilayer,abs2_u1_plus,abs2_u1_minus]),label=labels,size=(900,400),legend = :outerright, legendfontsize=10)
    # savefig(pl,string(p.path_plots,"Vint.png"))

    z_axis = (0:res-1)*p.L/res .- p.L/2

    res_plot = 500
    fig = CairoMakie.Figure(resolution = (floor(Int,res_plot*2),res_plot))

    funs = FFTW.fftshift.([v_int,v_plus,v_minus,abs2_u1_plus,abs2_u1_minus])#,v_bilayer
    labels = [L"V_{int, d}(z)" L"\frac{1}{\mid\Omega\mid} {\int_\Omega} V(x,z+d/2) d x" L"\frac{1}{\mid\Omega\mid} {\int_\Omega} V(x,z-d/2) d x" L"\int_\Omega \mid\Phi_1\mid^2(x,z+d/2) d x" L"\int_\Omega \mid\Phi_1\mid^2(x,z-d/2) d x"] #L"\frac{1}{\mid\Omega\mid^2} \int_{\Omega \times \Omega} V^{(2)}_{d,{y}}({x},z) d {x} d {y}"
    axs = [1,1,1,2,2]
    colors = [:purple,:blue,:cyan,:orange,:red]#:darkblue,

    ax1 = fig[1,1] = CairoMakie.Axis(fig, xlabel = "x (Bohr)", ylabel = "eV")
    ax2 = fig[1,1] = CairoMakie.Axis(fig, xlabel = "x (Bohr)", ylabel = "∅")
    CairoMakie.ylims!(ax2,(-0.5,1))
    for ax = [ax1,ax2]
        xlim = 15
        CairoMakie.xlims!(ax,(-xlim,xlim))
    end

    lins = []
    for i=1:length(funs)
        A = DataFrames.DataFrame(x=z_axis, y=funs[i])
        l = CairoMakie.lines!(axs[i]==1 ? ax1 : ax2, A.x, A.y,linestyle= axs[i]==1 ? nothing : [0.5, 1.0, 1.5, 2.5],color = colors[i],ygridvisible = axs[i]==1)
        push!(lins,l)
    end

    ax2.yaxisposition = :right
    ax2.yticklabelalign = (:left, :center)
    ax2.xticklabelsvisible = false
    ax2.xticklabelsvisible = false
    ax2.xlabelvisible = false

    CairoMakie.linkxaxes!(ax1,ax2)

    # leg = CairoMakie.Legend(fig[1,2], funs, labels)
    leg = CairoMakie.Legend(fig[1,2], lins,labels)

    # Save
    CairoMakie.save(p.path_plots*"Vint.pdf",fig)
    if p.export_plots_article
        # savefig(pl,string(p.path_plots_article,"Vint.pdf"))
        CairoMakie.save(p.path_plots_article*"Vint.pdf",fig)
    end
    px("Done plot Vint")
end

using LinearAlgebra, JLD, FFTW, Optim, LaTeXStrings
using CairoMakie
px = println
include("common_functions.jl")
include("misc/create_bm_pot.jl")

################## EffPotentials, parameters of graphene honeycomb structures

mutable struct EffPotentials
    # Lattice
    dim
    a1_micro; a2_micro
    a1_star_micro; a2_star_micro
    a; a1; a2; a1_star; a2_star
    lattice_2d
    graphene_lattice_orientation

    # Discretization numbers
    N; Nz
    N2d; N3d

    # Infinitesimal quantities
    dx; dS; dz; dv

    # Axis
    x_axis_cart
    k_axis; k_grid
    kz_axis

    # Volumes, surfaces, distances
    L; interlayer_distance
    cell_area; Vol
    sqi

    # Matrices
    R_four_2d
    M_four_2d

    # Dirac point quantities
    K_red
    q1; q2; q3
    q1_red; q2_red; q3_red
    m_q1
    gauge_param

    # Quantities computed and exported by graphene.jl
    v_f; u1_f; u2_f; u1v_f; u2v_f; prods_f; Vint_f
    v_dir; u1_dir; u2_dir; u1v_dir; u2v_dir; prods_dir; Vint_dir
    u1Vint_f; u2Vint_f
    vF # Fermi velocity
    non_local_coef
    non_local_φ1_f
    non_local_φ2_f
    shifts_atoms_red 

    # Effective potentials
    Σ # Σ = <<u_j, u_{j'}>>^+-
    𝕍_V; 𝕍_Vint; 𝕍; δ𝕍 # 𝕍_V = <<u_j, V u_{j'}>>^+-, 𝕍_Vint = <<u_j, Vint u_{j'}>>^+-, 𝕍 = 𝕍_V + 𝕍_Vint, δ𝕍 = 𝕍 - T_BM
    W_V_plus; Wplus_tot; W_V_minus # W_V_plus = << overline(u_j) u_{j'}, V >>^++, W_V_minus = << overline(u_j) u_{j'}, V >>^--, W_Vint_matrix = <u_j,Vint u_{j'}>
    Wminus_tot; W_Vint_matrix # Wplus_tot = W_V_plus + W_Vint_matrix, Wminus_tot = W_V_minus + W_Vint_matrix
    𝔸1; 𝔸2; 𝔹1; 𝔹2
    J𝔸1; J𝔸2
    W_non_local_plus; W_non_local_minus 
    add_non_local_W # ∈ [true,false]
    compute_Vint # ∈ [true,false] whether we take Vint into account
    T_BM # Bistritzer-MacDonald potential

    # Misc
    wAA # Bistritzer-MacDonald potential coupling constant

    # Plots
    plots_cutoff # Fourier cutoff of plotted functions
    plots_res # resolution in direct space, plots_res × plots_res
    plots_n_motifs # roughly the number of periods we see on plots
    root_path
    plots_x_axis_cart
    article_path
    plot_for_article

    function EffPotentials()
        p = new()
        p.dim = 2
        p.plots_n_motifs = 5
        p.plots_res = 20
        p.add_non_local_W = true
        p.article_path = "plots_article/"
        create_dir(p.article_path)
        p.vF = 0.380 # Fermi velocity, in Hartree
        p
    end
end

function init_EffPot(p)
    p.root_path = "effective_potentials/"
    create_dir(p.root_path)
    plots_a = p.a*p.plots_n_motifs
    plots_dx = plots_a/p.plots_res
    α = 1/2 # shift
    p.plots_x_axis_cart = (-α*plots_a:plots_dx:(1-α)*plots_a-plots_dx)
    p.m_q1 = -[1/3,1/3]
end

################## Imports graphene.jl quantities (u1, u2, V, Vint) and produces effective potentials without plotting and doing tests

function import_and_computes(N,Nz,gauge_param,compute_Vint,d)
    p = EffPotentials()
    p.compute_Vint = compute_Vint
    p.add_non_local_W = false
    p.interlayer_distance = d
    import_u1_u2_V_φ(N,Nz,gauge_param,p)
    import_Vint(p)
    init_EffPot(p)

    # Creates T_BM
    (m,α,Topt) = optimize_gauge_and_create_T_BM_with_α(p.compute_Vint ? "V" : "V_V",p)
    build_blocks_potentials(p)
    p.wAA = get_wAA_from_Eff(p)
    p.T_BM = T_BM_four(p.wAA,p.wAA,p)/p.sqi
    V = p.compute_Vint ? p.𝕍 : p.𝕍_V
    # px("Distance between V and optimized T_BM ",relative_distance_blocks(V,Topt))
    # px("Distance between V and wAA-computed T_BM ",relative_distance_blocks(V,p.T_BM))
    build_block_𝔸(p)
    p
end

function update_plots_resolution(n,p)
    p.plots_res = n
    init_EffPot(p)
end

# multiplies all potentials by γ
function multiply_potentials(γ,p)
    p.Σ *= γ
    p.𝕍_V *= γ; p.𝕍_Vint *= γ; p.𝕍 *= γ
    p.W_V_plus *= γ; p.Wplus_tot *= γ; p.W_V_minus *= γ; p.Wminus_tot *= γ; p.W_Vint_matrix *= γ
    p.𝔸1 *= γ; p.𝔸2 *= γ
    p.J𝔸1 *= γ; p.J𝔸2 *= γ
end

function BM(w,EffV,p) # assembles the Bistritzer-MacDonald potential in EffV
    EffV.wAA = w*ev_to_hartree
    T = T_BM_four(EffV.wAA,EffV.wAA,EffV)
    build_offdiag_V(T,p)
end

function reduce_N(p,N) # takes an EffPotentials and reduces the size of the plane-wave basis, eff.N -> N for the effective potentials
    p.N = N
    init_cell_infinitesimals(p)
    p.Σ = reduced_N_block(p.Σ,N)
    p.𝕍 = reduced_N_block(p.𝕍,N)
    p.𝕍_V = reduced_N_block(p.𝕍_V,N)
    p.𝕍_Vint = reduced_N_block(p.𝕍_Vint,N)
    p.W_V_plus = reduced_N_block(p.W_V_plus,N)
    p.Wplus_tot = reduced_N_block(p.Wplus_tot,N)
    p.W_V_minus = reduced_N_block(p.W_V_minus,N)
    p.Wminus_tot = reduced_N_block(p.Wminus_tot,N)
    p.𝔸1 = reduced_N_block(p.𝔸1,N)
    p.𝔸2 = reduced_N_block(p.𝔸2,N)
    p.J𝔸1 = reduced_N_block(p.J𝔸1,N)
    p.J𝔸2 = reduced_N_block(p.J𝔸2,N)
    p.W_non_local_plus = reduced_N_block(p.W_non_local_plus,N)
    p.W_non_local_minus = reduced_N_block(p.W_non_local_minus,N)
    p.T_BM = reduced_N_block(p.T_BM,N)
    # for B in [p.Σ,p.𝕍,p.𝕍_V,p.𝕍_Vint,p.W_V_plus,p.Wplus_tot,p.W_V_minus,p.Wminus_tot,p.𝔸1,p.𝔸2,p.J𝔸1,p.J𝔸2,p.W_non_local_plus,p.W_non_local_minus,p.T_BM]
    # B = reduced_N_block(B,N)
    # end
end

function get_low_Fourier_coefs(p) # prints the Fourier coefficients of lowest frequency, of the effective potentials
    px("Low Fourier coefficients of some effective potentials")
    V = p.compute_Vint ? p.𝕍 : p.𝕍_V
    px("V[1,1] "); print_low_fourier_modes(V[1],p;m=2)
    px("V[2,1] "); print_low_fourier_modes(V[3],p;m=2)
    px("W[1,1] "); print_low_fourier_modes(p.W_V_plus[1],p;m=2)
    px("W[2,1] "); print_low_fourier_modes(p.W_V_plus[3],p;m=2)
    px("T[1,1] "); print_low_fourier_modes(p.T_BM[1],p;m=2)
end

function print_infos_W(p) # to print some information about W
    px("### Infos on W")
    W = p.compute_Vint ? p.Wplus_tot : p.W_V_plus
    mean_W = mean_block(W,p)
    # means_W_V_minus = mean_block(p.W_V_minus,p)
    # @assert distance(means_W_V_minus,means_W_V_minus)<1e-4
    W_without_mean = add_cst_block(W,-sqrt(p.cell_area)*mean_W,p)
    px("Mean W_V_plus (meV) :")
    display(mean_block(p.W_V_plus,p)*1e3*hartree_to_ev)
    px("W_Vint matrix")
    display(p.W_Vint_matrix)
    px("Mean W_plus (meV) :")
    display(mean_block(p.Wplus_tot,p)*1e3*hartree_to_ev)
    px("### End infos on W")
    W_without_mean
end

################## Core: computation of effective potentials

function div_A(m,n,p)
    A = [2 1;1 2]; Ainv = (1/3)*[2 -1;-1 2]
    B = Ainv*[m;n]
    B1 = B[1]; B2 = B[2]
    mInt = myfloat2int(B1;warning=false,name="divA")
    nInt = myfloat2int(B2;warning=false,name="divA")
    if abs(mInt-B1)+abs(nInt-B2)>1e-5
        return (false,0,0)
    else
        return (true,mInt,nInt)
    end
end

# 2 × 2 matrix, magnetic ∈ {1,2}, f and g are in Fourier
# η ∈ {-1,+1}, (-i∇ + Q) ((g,f))^{+-}, Q in reduced
function build_magnetic(g,f,magnetic_term,p;η=1,Q=[0.0,0.0],coef_∇=1) 
    C = build_Cm(g,f,p;η=η)
    P = zeros(ComplexF64,p.N,p.N)
    q = Q[1]*p.a1_star .+ Q[2]*p.a2_star
    for m=1:p.N, n=1:p.N
        (m0,n0) = p.k_grid[m,n]
        K = coef_∇*(m0*p.a1_star .+ n0*p.a2_star) .+ q
        P[m,n] = K[magnetic_term]*C[m,n]
    end
    P
end

function compute_W_Vint_term(p) # matrix <u_j, Vint u_{j'}>
    M = zeros(ComplexF64,2,2)
    if p.compute_Vint
        for mx=1:p.N, my=1:p.N, nz1=1:p.N, nz2=1:p.N
            dmz = kz_inv(nz1-nz2,p)
            M[1,1] += conj(p.u1_f[mx,my,nz1])*p.u1_f[mx,my,nz2]*p.Vint_f[dmz]
            M[1,2] += conj(p.u1_f[mx,my,nz1])*p.u2_f[mx,my,nz2]*p.Vint_f[dmz]
            M[2,1] += conj(p.u2_f[mx,my,nz1])*p.u1_f[mx,my,nz2]*p.Vint_f[dmz]
            M[2,2] += conj(p.u2_f[mx,my,nz1])*p.u2_f[mx,my,nz2]*p.Vint_f[dmz]
        end
    end
    p.W_Vint_matrix = M/sqrt(p.L) # it is a direct space quantity !
end

function creates_δ𝕍(p)
    p.δ𝕍 = op_two_blocks((x,y)->x.-y,p.𝕍,T_BM_four(p.wAA/p.sqi,p.wAA/p.sqi,p))
end

################## Computation of blocks

function build_block_𝔸(p)
    p.𝔸1,p.𝔸2 = build_mag_block(p;Q=-p.q1_red)
    (p.J𝔸1,p.J𝔸2) = rot_block(π/2,p.𝔸1,p.𝔸2,p)
    # px("dist  ",distance(p.𝔸1[2][2,1],p.J𝔸2[2][2,1]))
    # px("dist  ",distance(p.𝔸2[2][end,2],-p.J𝔸1[2][end,2]))
end

function divAK(A1,A2,p;coef_∇=1,K=[0.0,0.0]) # A1 and A2 are 4×4 block functions gives (-i∇+K) ⋅ A
    M = []
    for g=1:4
        F = zeros(ComplexF64,p.N,p.N)
        for x=1:p.N, y=1:p.N
            q = (coef_∇*p.k_axis[x]+K[1])*p.a1_star + (coef_∇*p.k_axis[y]+K[2])*p.a2_star
            F[x,y] = q[1]*A1[g][x,y] + q[2]*A2[g][x,y]
        end
        push!(M,F)
    end
    M
end

function build_mag_block(p;Q=[0.0,0.0]) # (-i∇ + Q) ((u_j,u_{j'}))^{+-}, Q in reduced
    𝔸1 = [build_magnetic(p.u1_f,p.u1_f,1,p;Q=Q), build_magnetic(p.u1_f,p.u2_f,1,p;Q=Q),
          build_magnetic(p.u2_f,p.u1_f,1,p;Q=Q), build_magnetic(p.u2_f,p.u2_f,1,p;Q=Q)]

    𝔸2 = [build_magnetic(p.u1_f,p.u1_f,2,p;Q=Q), build_magnetic(p.u1_f,p.u2_f,2,p;Q=Q),
          build_magnetic(p.u2_f,p.u1_f,2,p;Q=Q), build_magnetic(p.u2_f,p.u2_f,2,p;Q=Q)]

    (𝔸1,𝔸2)
end

function test_div_JA(K1r,K2r,p) # tests that -i div J A = q_1 J A, q1 in cartesian coordinates
    div = divAK(p.J𝔸1,p.J𝔸2,p)
    qJA = divAK(p.J𝔸1,p.J𝔸2,p;coef_∇=0,K=K1r-K2r) #p.q1_red)
    px("-i div JA = q1 JA : ",relative_distance_blocks(div,qJA))
    q1 = k_red2cart(p.q1_red,p)
    K1 = k_red2cart(K1r,p)
    K2 = k_red2cart(K2r,p)
    px("q1 = K1-K2 ",distance(q1,K1-K2))
end

create_Σ(u1,u2,p) = [build_Cm(u1,u1,p), build_Cm(u1,u2,p),
                     build_Cm(u2,u1,p), build_Cm(u2,u2,p)]

create_V_V(u1,u2,p) = [build_Cm(p.u1v_f,p.u1_f,p), build_Cm(p.u1v_f,p.u2_f,p),
                       build_Cm(p.u2v_f,p.u1_f,p), build_Cm(p.u2v_f,p.u2_f,p)]

create_V_Vint(u1,u2,p) = [build_Cm(p.u1Vint_f,u1,p), build_Cm(p.u1Vint_f,u2,p),
                          build_Cm(p.u2Vint_f,u1,p), build_Cm(p.u2Vint_f,u2,p)]

create_V(u1,u2,p) = create_V_V(u1,u2,p) .+ create_V_Vint(u1,u2,p)

function build_blocks_potentials(p)
    # Σ
    p.Σ = create_Σ(p.u1_f,p.u2_f,p)

    # V
    p.𝕍_V = create_V_V(p.u1_f,p.u2_f,p)
    p.𝕍_Vint = create_V_Vint(p.u1_f,p.u2_f,p)
    p.𝕍 = p.𝕍_V .+ p.𝕍_Vint

    # W
    p.W_V_plus = [build_Cm(p.v_f,p.prods_f[1],p), build_Cm(p.v_f,p.prods_f[2],p),
                  build_Cm(p.v_f,p.prods_f[3],p), build_Cm(p.v_f,p.prods_f[4],p)]
    p.W_V_minus = [build_Cm(p.v_f,p.prods_f[1],p;η=-1), build_Cm(p.v_f,p.prods_f[2],p;η=-1),
                   build_Cm(p.v_f,p.prods_f[3],p;η=-1), build_Cm(p.v_f,p.prods_f[4],p;η=-1)]
    compute_W_Vint_term(p)
    compute_non_local_W(p)
    p.Wplus_tot = add_cst_block(p.W_V_plus,p.W_Vint_matrix/sqrt(p.cell_area),p)
    p.Wminus_tot = add_cst_block(p.W_V_minus,p.W_Vint_matrix/sqrt(p.cell_area),p)
    if p.add_non_local_W
        p.Wplus_tot .+= p.W_non_local_plus
        p.Wminus_tot .+= p.W_non_local_minus
    end
end

# direct space quantity
mean_block(V,p) = [V[i][1,1] for i=1:4]/sqrt(p.cell_area) 

function add_cst_block(B,cB,p) # B in Fourier so needs to add to the first component
    nB = deepcopy(B)
    for j=1:4
        nB[j][1,1] += cB[j]
    end
    nB
end

################## Effective terms extracted from the non local potential

function compute_non_local_F_term(η,j,s,p) # η ∈ {±}, j,s ∈ {1,2}
    F = zeros(ComplexF64,p.N,p.N)
    u = j==1 ? p.u1_f : p.u2_f
    φ = s==1 ? p.non_local_φ1_f : p.non_local_φ2_f
    # φ = s==1 ? p.u1_f : p.u2_f
    J = rotM(π/2)
    as_cart = p.shifts_atoms_red[s][1]*p.a1_micro + p.shifts_atoms_red[s][2]*p.a2_micro
    Jas = J*as_cart
    k_ar = [cis(-η*((2π/p.L)*p.interlayer_distance*mz)) for mz in p.kz_axis]
    for mix=1:p.N, miy=1:p.N
        (m0,n0) = p.k_grid[mix,miy]
        (m1,n1) = k_inv(η*m0,η*n0,p)
        k_cart = k_red2cart([m0,n0],p)
        F[mix,miy] = cis((1/2)*k_cart⋅Jas)*sum(k_ar[miz]*conj(φ[m1,n1,miz])*u[m1,n1,miz] for miz=1:p.Nz)
    end
    F
end

W_non_local_terms(η,i,j,p) = p.non_local_coef*(1/p.cell_area)*sum(cyclic_conv(conj_four(compute_non_local_F_term(η,i,s,p),p),compute_non_local_F_term(η,j,s,p),p.dS) for s=1:2)

# returns W_nl^{η}_{j,j'} in Fourier
W_non_local_plus_minus(η,p) = [W_non_local_terms(η,1,1,p), W_non_local_terms(η,1,2,p), W_non_local_terms(η,2,1,p), W_non_local_terms(η,2,2,p)] # η ∈ {±}

function compute_non_local_W(p)
    p.W_non_local_plus  = W_non_local_plus_minus( 1,p)
    p.W_non_local_minus = W_non_local_plus_minus(-1,p)
end

################## Operations on functions

function rot_A(θ,B1,B2,p) # applies R_θ to the vector [B1,B2], where Bj contains functions N×N
    R = rotM(θ)
    A1 = similar(B1); A2 = similar(B2)
    for K=1:p.N, P=1:p.N
        c = R*[B1[K,P],B2[K,P]]
        A1[K,P] = c[1]; A2[K,P] = c[2]
    end
    (A1,A2)
end

# a constant in Fourier space (typically, the component [1,1] of a Fourier function, which gives the average of the function) to the constant in direct space (resp. the average of the function in direct space)
Fourier_cst_to_direct_cst(x,p) = x/p.N2d

################## Operations on (2×2) block functions

norm_block(B,p) = sqrt(norm2(B[1],p) + norm2(B[2],p) + norm2(B[3],p) + norm2(B[4],p))
norm_block_potential(A1,A2,p) = sqrt(norm_block(A1,p)^2 + norm_block(A2,p)^2)
# sum(sqrt.(abs2.(A1[1]).+abs2.(A2[1])))+sum(sqrt.(abs2.(A1[2]).+abs2.(A2[2])))+sum(sqrt.(abs2.(A1[3]).+abs2.(A2[3])))+sum(sqrt.(abs2.(A1[4]).+abs2.(A2[4])))
app_block(map,B,p) = [map(B[1],p),map(B[2],p),map(B[3],p),map(B[4],p)]
op_two_blocks(op,A,B) = [op(A[i],B[i]) for i=1:4]
σ1_B_σ1(B) = [B[4],B[3],B[2],B[1]]
hermitian_block(B) = conj.([B[1],B[3],B[2],B[4]])
conj_block(B) = conj.([B[1],B[2],B[3],B[4]])
# hermitian_block(B) = [conj.(B[1]),conj.(B[3]),conj.(B[2]),conj.(B[4])]
U_B_U_star(B) = [B[1],cis(2π/3).*B[2],cis(4π/3).*B[3],B[4]]
anti_transpose(B) = [B[4],B[2],B[3],B[1]]
reduced_N_block(B,N) = [reduce_N_matrix(B[i],N) for i=1:4]

# Rotations on magnetic blocks, as a vector
function rot_block(θ,B1,B2,p)
    A1 = similar(B1); A2 = similar(B2)
    for j=1:4
        (A1[j],A2[j]) = rot_A(θ,B1[j],B2[j],p)
    end
    (A1,A2)
end

function weight(M,α,β) # M in matrix form, applies weights α and β
    N = size(M[1,1],1)
    m = zeros(ComplexF64,N,N)
    S = [deepcopy(m) for i=1:2, j=1:2]
    S[1,1] = α*M[1,1]; S[1,2] = β*M[1,2]; S[2,1] = β*M[2,1]; S[2,2] = α*M[2,2]
    S
end

# from potentials in matrix form to potentials in vector form
mat2lin(M) = [M[1,1],M[1,2],M[2,1],M[2,2]]

# from potentials in vector form to potentials in matrix form
function lin2mat(M)
    N = size(M[1],1)
    m = zeros(ComplexF64,N,N)
    T = [deepcopy(m) for i=1:2, j=1:2]
    T[1,1] = M[1]; T[1,2] = M[2]; T[2,1] = M[3]; T[2,2] = M[4]
    T
end

################## Comparison of functions

function compare(u,v)
    α0 = [1.0]
    f(α) = distance(α[1]*u,v)
    res = optimize(f, α0)
    mz = Optim.minimizer(res)
    (minimum(res),mz)
end

function compare_blocks(A,B,p)
    for j=1:4
        (m,mz) = compare(A[j],B[j])
        px("Distance block ",j," ",m," minimizer ",mz)
    end
end

function compare_one_block(A,n,p) # A is the function, not a 4×4 block
    n1 = norm(A)
    function dist(α)
        T = T_BM_four(α[1],α[2],p)
        distance(T[n],A)
    end
    α0 = [1.0,1.0]
    res = optimize(dist, α0)
    m = minimum(res)
    α = Optim.minimizer(res)
    (m,α)
end

function compare_to_BM(A,p)
    function dist(α)
        T = T_BM_four(α[1],α[1],p)
        relative_distance_blocks(A,T)
    end
    res = optimize(dist, [1.0])
    m = minimum(res)
    α = Optim.minimizer(res)[1]
    T = T_BM_four(α,α,p)
    # px("Creating TBM from effectgive potential : distance to BM ",m," with coefs (α,β)=(",α,",",α,")")
    (m,α,T)
end

function compare_to_BM_infos(A,p,name)
    (m,α,T) = compare_to_BM(A,p)
    d1 = distance(T[1],A[1])
    d2 = distance(T[2],A[2])
    px("Distances blocks 1 and 2 between ",name," and optimally rescaled T_BM: ",d1," ",d2," obtained with α=",α)
end

get_create(pot) = pot == "V" ? create_V : (pot == "V_V" ? create_V_V : create_Σ)

function optimize_gauge_and_create_T_BM_with_α(pot,p)
    @assert pot in ["V","V_V","Σ"]
    create = get_create(pot)
    A = create(p.u1_f,p.u2_f,p)
    (m,α,T) = compare_to_BM(A,p)
    (m,α,T)
    # px("MIN with only α ",m," with ",α)
end

################## Get wAA wAB

print_wAA(p) = px("wAA",p.compute_Vint ? "" : "_v"," = ",p.wAA*hartree_to_ev*1e3," meV")

function get_wAA_from_Eff(p)
    px("##### wAA from effective potentials")
    wAA_v = real(p.𝕍_V[1][1,1])*p.sqi; wAA = wAA_v
    c = 1e3*hartree_to_ev
    px("wAA_v = ",c*wAA_v)
    if p.compute_Vint
        wAA_vint = real(p.𝕍_Vint[1][1,1])*p.sqi
        wAA = real(p.𝕍[1][1,1])*p.sqi
        px("wAA_vint = ",c*wAA_vint," meV")
        px("wAA = ",c*wAA," meV")
    end
    px("#####")
    wAA
end

function analyze(x)
    r,θ = polar(x)
    c = cis(θ)
    ω = cis(2π/3)
    res = "?"
    lim = 5e-2
    if true # First method
        pos = cis.(vcat([i*2π/6 for i=0:5],[π/2,-π/2,sqrt(3)/2]))
        pos_str = ["1","-ω^2","ω","-1","ω^2","-ω","i","-i","sqrt(3)/2"]
    else # Second method
        n = (3*2*5*7*11)^2
        pos = [cis(i*2π/n) for i=0:5]
        pos_str = [string(i,"*2π/",n) for i=0:n-1]
    end
    for j=1:length(pos)
        if abs(c-pos[j])<lim
            res = pos_str[j]
        end
    end
    r,res,θ
end

function print_low_fourier_modes(v,p,c=1;m=1)
    for mix=1:p.N, miy=1:p.N
        mx,my = p.k_axis[mix],p.k_axis[miy]
        if abs(mx) ≤ m && abs(my) ≤ m
            y = v[mix,miy]
            r,x,θ = analyze(y)
            px("mx,my= ",mx,",",my," : ",x!="?" ? x : x," ",r)
        end
    end
end

function rescale_A(V,p;shift=false) # M(x) = V(3x)
    M = zeros(ComplexF64,p.N,p.N)
    # (mK,nK) = Tuple(Int.(3*p.K_red))
    for mix=1:p.N, miy=1:p.N
        (m0,n0) = k_axis(mix,miy,p)
        if shift m0 -= 1; n0 -= 1 end
        (B,m1,n1) = div_A(m0,n0,p)
        if !B
            M[mix,miy] = 0
        else
            m2,n2 = k_inv(m1,n1,p)
            M[mix,miy] = V[m2,n2]
        end
    end
    M
end

rescale_A_block(V,p;shift=false) = [rescale_A(V[i],p;shift=shift) for i=1:4]

################## Symmetry information

function info_mirror_symmetry(p)
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
end

function info_particle_hole_symmetry(p)
    px("\nTests particle-hole symmetry")
    test_particle_hole_block(p.T_BM,p;name="T")
    test_particle_hole_block(p.𝕍,p;name="V")
    test_particle_hole_block_W(p.W_V_plus,p.W_V_minus,p;name="W_V")
    test_particle_hole_block_W(p.W_non_local_plus,p.W_non_local_minus,p;name="Wnl")
    test_particle_hole_block(p.Σ,p;name="Σ")
    test_particle_hole_block(p.𝔸1,p;name="A1")
    test_particle_hole_block(p.𝔸2,p;name="A2")
end

function info_PT_symmetry(p)
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
end     

function info_translation_symmetry(p)
    px("\nTests translation symmetry")
    test_one_translation_symmetry(p.𝕍_V,p;name="𝕍_V")
    test_one_translation_symmetry(p.Σ,p;name="Σ")
    test_one_translation_symmetry(p.𝔸1,p;name="𝔸1")
    test_one_translation_symmetry(p.𝔸2,p;name="𝔸2")
end

function info_equality_some_blocks_symmetry(p)
    px("\nTests equality inside blocks")
    test_equality_all_blocks(p.T_BM,p;name="T")
    test_equality_all_blocks(p.Wplus_tot,p;name="W")
    test_equality_all_blocks(p.𝕍,p;name="V")
    test_equality_all_blocks(p.Σ,p;name="Σ")
    test_equality_all_blocks(p.𝔸1,p;name="A1")
    test_equality_all_blocks(p.𝔸2,p;name="A2")
    # px("Equality blocks V and V_minus ",relative_distance_blocks(V,V_minus))
end

################## Symmetry tests

relative_distance_blocks(B,C) = sum(distance(B[i],C[i]) for i=1:4)/4

function test_one_translation_symmetry(B,p;name="B")
    px(name,"_{11}(x+(1/3)(a1-a2)) = ",p.gauge_param==1 ? "-" : "",name,"_{12}(x) ",distance(translation_interpolation(B[1],-[1/3,-1/3],p),-p.gauge_param*B[2]))
end

function test_particle_hole_block(B,p;name="B")
    PB_four = hermitian_block(B) # parity ∘ conj in direct becomes just conj in Fourier
    px("Test ",name,"(-x)^* = ",name,"(x) ",relative_distance_blocks(B,PB_four))
end

function test_particle_hole_block_W(W_plus,W_minus,p;name="W")
    P_Wplus = app_block(parity_four,W_plus,p)
    px("Test ",name,"+(-x) = ",name,"-(x) ",relative_distance_blocks(W_minus,P_Wplus))
end

function test_sym_Wplus_Wminus(p)
    Wm = p.W_V_minus
    TW = [Wm[4],Wm[2],Wm[3],Wm[1]]
    d = relative_distance_blocks(p.W_V_plus,TW)
    px("Test antitranspose(W_V_minus) = W_V_plus : ",d)
end

function test_PT_block_direct(B,p;name="B")
    B_direct = [myifft(B[i],p.Vol) for i=1:length(B)]
    σ1Bσ1 = σ1_B_σ1(B_direct)
    symB = conj.(app_block(parity_x,σ1Bσ1,p))
    px("Test σ1 conj(",name,")(-x) σ1 = ",name,"(x) ",relative_distance_blocks(B_direct,symB))
end

function test_PT_block(B,p;name="B")
    σ1Bσ1 = σ1_B_σ1(B)
    symB = conj_block(σ1Bσ1)
    px("Test σ1 conj(",name,")(-x) σ1 = ",name,"(x) ",relative_distance_blocks(B,symB))
end

function test_R_magnetic_block(B1,B2,p;name="B")
    RB1 = app_block(R_four,B1,p)
    RB2 = app_block(R_four,B2,p)
    U_B1_Ustar = U_B_U_star(B1)
    U_B2_Ustar = U_B_U_star(B2)
    (R_U_B1_Ustar,R_U_B2_Ustar) = rot_block(-2π/3,U_B1_Ustar,U_B2_Ustar,p)
    d = relative_distance_blocks(R_U_B1_Ustar,RB1) + relative_distance_blocks(R_U_B2_Ustar,RB2)
    px("Test R ",name," = R_{-2π/3 on vector} U",name,"U* ",d)
end

function test_mirror_block(B,p;name="B",herm=false)
    HB = B
    if herm
        HB = hermitian_block(B)
    end
    σ1Bσ1 = σ1_B_σ1(HB)
    symB = app_block(σ1_four,σ1Bσ1,p)
    px("Test σ1 ",name,(herm ? "*" : ""),"(x1,-x2) σ1 = ",name,"(x) ",relative_distance_blocks(B,symB))
end

function test_build_potential_direct(g,f,p) # by P(x) = ∑_m Cm e^{i2πxJ^*m}, used to test because it's much heavier than building by Fourier. PERIOD L, NOT L/2 !
    C = build_Cm(g,f,p)
    P = zeros(ComplexF64,p.N,p.N)
    calJ_star = [1 -2;
                 2 -1]
    calJ_m = [calJ_star*[p.k_grid[m,n][1];p.k_grid[m,n][2]] for m=1:p.N, n=1:p.N]
    for x=1:p.N, y=1:p.N
        expos = [exp(im*2π*(p.x_axis_cart[x]*calJ_m[m1,m2][1]+p.x_axis_cart[y]*calJ_m[m1,m2][2])) for m1=1:p.N, m2=1:p.N]
        P[x,y] = sum(C.*expos)
    end
    P
end

function test_equality_blocks_interm(B,p)
    c = sum(abs.(B[1]))
    px("(1,1)-(1,2)",sum(abs.(B[1].-B[2]))/c," ",
       "(1,1)-(2,1)",sum(abs.(B[1].-B[3]))/c," ",
       "(1,1)-(2,2)",sum(abs.(B[1].-B[4]))/c," ",
       "(1,2)-(2,1)",sum(abs.(B[2].-B[3]))/c," ")
end

function test_equality_all_blocks(B,p;name="")
    px("Test equality of blocks of ",name)
    D = [myifft(B[i],p.Vol) for i=1:length(B)]
    px("In direct")
    test_equality_blocks_interm(D,p)
    px("In Fourier")
    test_equality_blocks_interm(B,p)
end

function test_block_hermitianity(C,p;name="")
    B = [myifft(C[i],p.Vol) for i=1:length(C)]
    c = 0; T = 0
    for x=1:p.N, y=1:p.N
        h = [B[1][x,y] B[2][x,y]; B[3][x,y] B[4][x,y]]
        c += sum(abs.(h.-h'))
        T += sum(abs.(h))
    end
    px("Test block hermitianity ",name," ",c/T)
end


################## Import functions

function import_u1_u2_V_φ(N,Nz,gauge_param,p)
    p.N = N; p.Nz = Nz; p.gauge_param = gauge_param
    marker_gauge = p.gauge_param == 1 ? "p" : "m"
    path = "apply_graphene_outputs/exported_functions/"
    f = string(path,"N",p.N,"_Nz",p.Nz,"_g",marker_gauge,"_u1_u2_V_nonLoc.jld")

    p.a = load(f,"a"); p.L = load(f,"L")
    @assert p.gauge_param == load(f,"gauge_param")

    init_cell_vectors(p;moire=true)
    init_cell_infinitesimals(p)

    # Treats v
    p.v_f = load(f,"v_f")
    p.v_dir = myifft(p.v_f,p.Vol)
    substract_by_far_value(p.v_dir,p)
    p.v_f = myfft(p.v_dir,p.Vol)

    p.u1_f = load(f,"u1_f")
    p.u2_f = load(f,"u2_f")
    p.non_local_coef = load(f,"non_local_coef")
    p.non_local_φ1_f = load(f,"φ1_f")
    p.non_local_φ2_f = load(f,"φ2_f")
    p.shifts_atoms_red = load(f,"shifts_atoms")

    # Builds products from imports
    p.u1_dir = myifft(p.u1_f,p.Vol)
    p.u2_dir = myifft(p.u2_f,p.Vol)

    p.u1v_dir = p.v_dir.*p.u1_dir
    p.u2v_dir = p.v_dir.*p.u2_dir

    p.u1v_f = myfft(p.u1v_dir,p.Vol)
    p.u2v_f = myfft(p.u2v_dir,p.Vol)

    p.prods_dir = [abs2.(p.u1_dir), conj.(p.u1_dir).*p.u2_dir, conj.(p.u2_dir).*p.u1_dir, abs2.(p.u2_dir)]
    p.prods_f = [myfft(p.prods_dir[i],p.Vol) for i=1:length(p.prods_dir)]

    p.u1Vint_f = zeros(ComplexF64,p.N,p.N)
    p.u2Vint_f = zeros(ComplexF64,p.N,p.N)
end

function import_Vint(p)
    if p.compute_Vint
        marker_gauge = p.gauge_param == 1 ? "p" : "m"
        path = "apply_graphene_outputs/exported_functions/"
        f = string(path,"N",p.N,"_Nz",p.Nz,"_g",marker_gauge,"_d",p.interlayer_distance,"_Vint.jld")
        d = load(f,"d"); a = load(f,"a"); L = load(f,"L")
        # px("L ",p.L," ",L)
        @assert a==p.a 
        @assert L==p.L
        @assert p.interlayer_distance==d

        Vint_f = load(f,"Vint_f")
        p.Vint_f = zeros(ComplexF64,p.N,p.N,p.Nz)
        p.Vint_f[1,1,:] = sqrt(p.cell_area)*Vint_f
        p.Vint_dir = myifft(p.Vint_f,p.Vol)

        if p.compute_Vint
            p.u1Vint_f = myfft(p.u1_dir.*p.Vint_dir,p.Vol)
            p.u2Vint_f = myfft(p.u2_dir.*p.Vint_dir,p.Vol)
        end
    end
end

################## Plot functions

# creates the function of reduced direct space from the array in reduced Fourier space
function red_arr2red_fun(ϕ_four_red,p,k_red_shift=[0.0,0.0])
    k1 = k_red_shift[1]; k2 = k_red_shift[2]
    a(x,y) = 0
    for i=1:p.N
        ki = p.k_axis[i]
        if abs(ki) <= p.plots_cutoff
            for j=1:p.N
                kj = p.k_axis[j]
                if abs(kj) <= p.plots_cutoff
                    c(x,y) = (ϕ_four_red[i,j] * cis(2π*((ki+k1)*x+(kj+k2)*y)))/sqrt(p.cell_area)
                    a = a + c
                end
            end
        end
    end
    a
end

function red_arr2cart_fun(ϕ_four_red,p,k_red_shift=[0.0,0.0])
    k1 = k_red_shift[1]; k2 = k_red_shift[2]
    a(x,y) = 0
    for i=1:p.N
        ki = p.k_axis[i]
        if abs(ki) <= p.plots_cutoff
            for j=1:p.N
                kj = p.k_axis[j]
                if abs(kj) <= p.plots_cutoff
                    ma_star = (ki+k1)*p.a1_star+(kj+k2)*p.a2_star
                    c(x,y) = (ϕ_four_red[i,j] * cis(ma_star⋅[x,y]))/sqrt(p.cell_area)
                    a = a + c
                end
            end
        end
    end
    a
end

# from function in reduced direct space to function in cartesian direct space
function red2cart_function(f,p)
    cart2red = inv(hcat(p.a1,p.a2))
    function g(x,y)
        v = cart2red*[x,y]
        f(v[1],v[2])
    end
    g
end

function eval_fun_to_plot(f_four,fun,n_motifs,p;k_red_shift=[0,0.0])
    # Computes function in cartesian space
    # fu = red_arr2red_fun(f_four,p,k_red_shift)
    # ψ2 = red2cart_function(fu,p)
    f = red_arr2cart_fun(f_four,p,k_red_shift)

    # Evaluates
    res = length(p.plots_x_axis_cart)
    fun.([f(p.plots_x_axis_cart[i],p.plots_x_axis_cart[j]) for i=1:res, j=1:res])
    # fun.([f(i/res,j/res) for i=0:res-1, j=0:res-1])
end

# B is a 2 × 2 matrix of potentials
# from array of Fourier coefficients to plot in direct cartesian space
function plot_block_cart(B_four,p;title="plot_full",article=false)
    path = string(p.root_path,"plots_potentials_cartesian_N",p.N,"_Nz",p.Nz,"/")
    create_dir(path)
    funs = [real,imag,abs]; titles = ["real","imag","abs"]
    expo = -1
    for I=1:3
        h = []
        for m=1:4
            ψ_ar = eval_fun_to_plot(B_four[m],funs[I],p.plots_n_motifs,p)
            if expo == -1
                if maximum(abs.(ψ_ar)) < 1e-6
                    expo = 0
                else
                    expo = floor(Int,log10(maximum(abs.(ψ_ar)))) 
                end
            end
            hm = Plots.heatmap(p.plots_x_axis_cart,p.plots_x_axis_cart,ψ_ar*10^(-expo),size=(300,200),colorbar_title=latexstring("10^{$(expo)}"))#,colorbar_titlefontrotation=180,colorbar_titlefontvalign=:top)

            # hm = heatmap(ψ_ar,aspect_ratio=:equal)
            push!(h,hm)
        end
        size = 1000
        pl = Plots.plot(h...,layout=(2,2),size=(1300,1000),legend=false)
        titl = string(title,"_",titles[I],"_cart")
        savefig(pl,string(path,titl,".png"))
        # if article && p.plot_for_article
        # savefig(pl,string(p.article_path,titl,".pdf"))
        # end
        px("Plot of ",title," ",titles[I]," in cartesian coords, done")
    end
end

# other_magnetic_block is the additional magnetic block, in this case it plots |B1|^2+|B2|^2
function plot_magnetic_block_cart(B1_four,B2_four,p;title="plot_full",article=false) 
    h = []
    path = string(p.root_path,"plots_potentials_cartesian_N",p.N,"_Nz",p.Nz,"/")
    create_dir(path)

    for m=1:4
        ψ_ar = eval_fun_to_plot(B1_four[m],abs2,p.plots_n_motifs,p)
        ψ_ar .+= eval_fun_to_plot(B2_four[m],abs2,p.plots_n_motifs,p)
        ψ_ar = sqrt.(ψ_ar)
        hm = Plots.heatmap(p.plots_x_axis_cart,p.plots_x_axis_cart,ψ_ar,size=(300,200))
        # hm = Plots.heatmap(ψ_ar,aspect_ratio=:equal)
        push!(h,hm)
    end
    size = 1000
    pl = Plots.plot(h...,layout=(2,2),size=(1300,1000),legend=false)
    titl = string("abs_",title,"_cart.png")
    savefig(pl,string(path,title))
    if article && p.plot_for_article
        savefig(pl,string(p.article_path,titl))
    end
    px("Plot of |",title,"|^2 in cartesian coords, done")
end

function plot_block_reduced(B,p;title="plot_full")
    four = [true,false]
    funs = [real,imag,abs]; titles = ["real","imag","abs"]
    path = string(p.root_path,"plots_potentials_reduced_N",p.N,"_Nz",p.Nz,"/")
    create_dir(path)
    for fo=1:2, I=1:3
        h = []
        for m=1:4
            a = funs[I].(four[fo] ? B[m] : myifft(B[m],p.Vol))
            hm = Plots.heatmap(a,size=(200,200),aspect_ratio=:equal)
            push!(h,hm)
        end
        size = 1000
        pl = Plots.plot(h...,layout=(2,2),size=(1300,1000),legend=false)
        savefig(pl,string(path,title,"_",titles[I],"_",four[fo] ? "four" : "dir",".png"))
        # px("Plot of ",title," ",titles[I]," done")
    end
end

function eval_blocks(B_four,funs,p;k_red_shift=[0,0.0])
    ars = []
    for I=1:length(funs)
        h = []
        for m=1:4
            ψ = eval_fun_to_plot(B_four[m],funs[I],p.plots_n_motifs,p;k_red_shift=k_red_shift)
            push!(h,ψ)
        end
        push!(ars,h)
    end
    ars
end

function plot_block_article(B_four,p;title="plot_full",other_block=-1,k_red_shift=[0.0,0.0],meV=true,coef=1,vertical_bar=false) # coef applies a coef so that we draw coef*B
    funs = [real,imag,abs]; titles = ["real","imag","abs"]
    n = length(funs)
    ars = eval_blocks(B_four,funs,p;k_red_shift=k_red_shift)*coef
    if other_block!=-1 # case magnetic term with another block
        ars2 = eval_blocks(other_block,funs,p)*coef
        op(x,y) = sqrt.(abs2.(x) .+ abs2.(y))
        for i=1:3
            ars[i] = op_two_blocks(op,ars[i],ars2[i])
        end
    end
    # meV units
    if meV
        ars = ars*1e3*hartree_to_ev
        if other_block!=-1
            ars2 = ars2*1e3*hartree_to_ev
        end
    end
    # Computes common colorbar
    m = minimum(minimum(ars[I][j]) for I=1:n, j=1:4)
    M = maximum(maximum(ars[I][j]) for I=1:n, j=1:4)
    joint_limits = (m,M)

    ismag = other_block!=-1
    # Plots functions
    res_bar = vertical_bar ? 700 : 1200
    X,Y = vertical_bar ? (floor(Int,res_bar/3),floor(Int,1.0*res_bar)) : (floor(Int,1.2*res_bar),floor(Int,res_bar/4))
    fig_colors = CairoMakie.Figure(resolution=(X,Y),fontsize = vertical_bar ? 22 : 35) # creates colorbar
    # colormap ∈ [:heat,:viridis]
    clm = :Spectral
    # clm = :linear_bmy_10_95_c78_n256
    hm(fi,f) = Makie.heatmap(fi,p.plots_x_axis_cart,p.plots_x_axis_cart,f,colormap=clm,colorrange=joint_limits)

    res = 700
    for I=1:n
        fig = CairoMakie.Figure(resolution=(res,res))
        ff1,ax1 = hm(fig[1,1],ars[I][1])
        ff2,ax2 = hm(fig[1,2],ars[I][2])
        ff3,ax3 = hm(fig[2,1],ars[I][3])
        ff4,ax4 = hm(fig[2,2],ars[I][4])

        for ff in [ff1,ff2,ff3,ff4]
            CairoMakie.hidedecorations!(ff, grid = false)
        end
        fact = 1
        CairoMakie.arrows!(ff1,[0,0],[0,0],fact*[p.a1[1],p.a2[1]],fact*[p.a1[2],p.a2[2]], arrowsize = 10)
        CairoMakie.arrows!(ff2,[0,0],[0,0],fact*[p.a1[1],p.a2[1]],fact*[p.a1[2],p.a2[2]], arrowsize = 10)
        # arrows!(ff1,[0],[0],[1],[4], arrowsize = 10)
        sh = [1,0]
        if funs[I]==abs
            CairoMakie.text!(ff1,[L"\epsilon_{\theta}^{-1} a_{1,M}",L"\epsilon_{\theta}^{-1} a_{2,M}"],position = Tuple.(fact*[p.a1.-1.3*sh,p.a2.-7*sh]),textsize=40)
        end

        titl = string(title,"_",titles[I],"_cart")
        CairoMakie.save(string(p.article_path,titl,".pdf"),fig)
        CairoMakie.Colorbar(fig_colors[1,1],ax1,vertical=vertical_bar,flipaxis=vertical_bar,label=meV ? "meV" : "") # records colorbar. flipaxis to have ticks down the colormap
    end

    # Saves colorbar
    titl = string(p.article_path,title,"_colorbar.pdf")
    CairoMakie.save(titl,fig_colors)
    px("Done ploting article ",title," in meV")
end

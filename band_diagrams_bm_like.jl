include("misc/lobpcg.jl")
include("common_functions.jl")
include("misc/create_bm_pot.jl")
# include("misc/plot.jl")
include("effective_potentials.jl")
using LinearAlgebra, JLD, FFTW, Plots, LaTeXStrings
# using MKL

################## Parameters

mutable struct Basis
    # Lattice
    dim
    a; a1; a2; a1_star; a2_star
    a1_micro; a2_micro
    a1_star_micro; a2_star_micro
    lattice_2d
    graphene_lattice_orientation

    # Discretization numbers
    N; Nz

    # Infinitesimal quantities
    dx; dS; dv; dz

    # Axis
    x_axis_cart
    k_axis; k_grid
    kz_axis

    # Volumes, surfaces, distances
    cell_area; Vol; L
    sqi

    # Matrices
    M_four_2d
    R_four_2d

    # Dirac point quantities
    K_red
    q1; q2; q3; q1_red; q2_red; q3_red
    m_q1
    vF
    K1; K2

    # Objects for the coordinates of the Hamiltonian
    N2; Mfull
    k_grid_lin
    Li_tot; Li_k; Ci_tot; Ci_k; k2lin
    l; l1; l2; n_l

    # Other objects
    S # part_hole operator
    ISΣ # (1+S)^{-1/2}
    solver

    # Bands diagram
    Klist
    Klist_names

    # Parameters for plots
    root_path
    energy_center
    energy_amplitude
    folder_plots_bands
    folder_plots_matrices
    resolution_bands
    energy_unit_plots # ∈ ["eV","Hartree"]
    plots_article # ∈ ["false","true"]
    path_plots_article
    path_bandwidths
    function Basis()
        p = new()
        p.vF = 0.380
        p.plots_article = false
        p.path_plots_article = "plots_article/"
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
    number_of_states = 4
    p.N2 = p.N^2
    p.Mfull = number_of_states*p.N2

    ref = zeros(number_of_states,p.N,p.N); ref_k = zeros(p.N,p.N)
    p.Li_tot = LinearIndices(ref);    p.Li_k = LinearIndices(ref_k)
    p.Ci_tot = CartesianIndices(ref); p.Ci_k = CartesianIndices(ref_k)
    init_klist2(p)

    p.ISΣ = -1 # matrix (1+S_Σ)^(-1/2)
    p.solver = "LOBPCG"

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

# Creates the list of squares of momentums
function init_klist2(p)
    p.k2lin = []
    for c_lin=1:p.Mfull
        (i,pix,piy) = lin2coord(c_lin,p)
        push!(p.k2lin,p.k_axis[pix]^2 + p.k_axis[piy]^2)
    end
end

######################### Coordinates change

function lin2coord(c_lin,p)
    ci = p.Ci_tot[c_lin]
    (ci[1],ci[2],ci[3]) # (i,pix,piy)
end

k_axis(pix,piy,p) = (p.k_axis[pix],p.k_axis[piy])

######################### Coordinates change

# from the solution of LOBPCG ψ_lin[4*p.N^2] to 4 vectors containing the fourier transforms
function lin2four(ψ_lin,p) 
    ψ_four = fill2d(zeros(ComplexF64,p.N,p.N),4)
    for lin=1:p.Mfull
        (i,pix,piy) = lin2coord(lin,p)
        ψ_four[i][pix,piy] = ψ_lin[lin]
    end
    ψ_four
end

######################### Fill matrices

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

function σK(H,i,I,q,v,p;c=1,J=false)
    x = X(i,I,p)
    Q = k_red2cart(q,p)
    if J
        Q = [-Q[2], Q[1]]
    end
    H[x,x+1] = c*(v*Q[1] - im*Q[2])
    H[x+1,x] = c*(v*Q[1] + im*Q[2])
end

function mΔK(H,i,I,q,p)
    x = X(i,I,p)
    Q = k_red2cart(q,p)
    n = norm(Q)^2
    H[x,x] = n
    H[x+1,x+1] = n
end

######################### Derivation operators

# Creates [0                 (-i div) Σ(-i∇)]
#         [(-i div) Σ^*(-i∇)               0]

function offdiag_div_Σ_∇(Σ,K,p;K1=p.K1,K2=p.K2,name="",test=false)
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

# Creates [0          𝔸⋅(-i∇+k)]
#         [𝔸*⋅(-i∇+k)         0]

function offdiag_A_k(A1,A2,K,p;K1=p.K1,K2=p.K2,name="",test=false)
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
            end
        end
    end
    # H = H+H'
    if test
        test_hermitianity(H,string(name,"A"))
    end
    H
end

# Creates [coef_1*σ(-i∇+k-K1)           0]
#         [0                  σ(-i∇+k-K2)]

function Dirac_k(K,p;valley=1,K1=p.K1,K2=p.K2,coef_1=1,J=false,test=false)
    H = zeros(ComplexF64,4*p.N2, 4*p.N2)
    for i=1:p.N2
        n1,n2 = p.k_grid_lin[i]
        KG = K + [n1,n2]
        σK(H,i,1,KG-K1,valley,p;c=coef_1,J=J)
        σK(H,i,2,KG-K2,valley,p;J=J)
    end
    if test
        test_hermitianity(H,"σ(-i∇)")
    end
    H
end


#  Creates [(-i∇+k-K1)^2            0]
#          [0            (-i∇+k-K2)^2]

function ondiag_mΔ_k(K,p;K1=p.K1,K2=p.K2,test=false)
    H = zeros(ComplexF64,4*p.N2, 4*p.N2)
    for i=1:p.N2
        n1,n2 = p.k_grid_lin[i]
        KG = K + [n1,n2]
        mΔK(H,i,1,KG-K1,p)
        mΔK(H,i,2,KG-K2,p)
    end
    if test
        test_hermitianity(H,"Δ")
    end
    H
end

######################### Electric potential operators

# Creates [0  𝕍]
#         [𝕍* 0]

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

## Creates [𝕎^+   0]
#          [0   𝕎^-]

function build_ondiag_W(Wplus,Wminus,p;test=false)
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


######################### Add weights

# adds weights V = [a b ; c d] -> [αa βb ; βc αd]
function weights_off_diag_matrix(V,α,β,p)
    @assert abs(imag(α)) + abs(imag(β)) < 1e-10
    (α-1)*p.Iα.*V .+ (β-1)*p.Iβ.*V .+ V
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
            h = antiherm(HvK)
            if h>1e-4 px("Be careful, H not exactly Hermitian : ",h) end
            HvK_herm = Hermitian((HvK+HvK')/2)
            (E,Xf) = solve_one(HvK_herm,p,X)
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

fermi_label(p) = floor(Int,p.n_l/2) # the label of the lower Fermi band, the upper one is fermi_label +1

function bandwidth_bandgap_fermivelocity(σ,p)
    nmid = fermi_label(p)
    # Bandwidth
    bw = maximum(σ[:,nmid+1])-minimum(σ[:,nmid])

    # Bandgap
    bg = max(0,minimum(σ[:,nmid+2])-maximum(σ[:,nmid+1]))

    # Fermi velocity
    K1 = k_red2cart(p.Klist[1],p)
    K2 = k_red2cart(p.Klist[2],p)
    dk = norm(K1-K2)/p.resolution_bands
    nmid = fermi_label(p)
    fv = abs( (σ[2,nmid+1] - σ[1,nmid+1])/dk )

    (bw,bg,fv)
end

function coef_plot_meV(θ,p)
    θrad = (π/180)*θ
    εθ = 2*sin.(θrad/2)
    hartree_to_ev*1e3*εθ
end



function plot_bandwidths_bandgaps_fermivelocities(θs,bw,bg,fv,p;def_ticks=true,name="thetas")
    labels = ["Fermi velocity","Band gap","Bandwidth"]
    # filenames = ["bandwidth","bandgap","fermi_velocity"]
    quantities = [fv,bg,bw]
    fig = CairoMakie.Figure(resolution=(600,1000))
    for fi=1:3
        ylab = labels[fi]
        if fi!=1
            ylab = string(ylab," (meV)")
        end
        ax = CairoMakie.Axis(fig[fi,1], xlabel = "θ (degrees)", ylabel=ylab)
        if fi in [1,2]
            hidexdecorations!(ax, grid = false)
        end
        if def_ticks ax.xticks = (θs[1]: 0.1 :θs[end]) end
        CairoMakie.xlims!(ax,θs[1],θs[end])

        colors = [:red,:black,:blue]
        n = length(θs)
        ymax = 0

        function cf(fi,i)
            coef = 1
            if fi in [2,3]
                coef = coef_plot_meV(θs[i],p)
            elseif fi==1
                coef = 1/0.380 # fermi velocity of the monolayer, in Hartree
            end
            coef
        end

        for j=1:3
            ys = [quantities[fi][j][i]*cf(fi,i) for i=1:n]
            if ymax < maximum(ys) ymax = maximum(ys) end
            points = [CairoMakie.Point2f0(θs[i],ys[i]) for i=1:n]
            CairoMakie.lines!(points,color=colors[j])
        end
        CairoMakie.ylims!(ax,0,max(ymax,0.01)) # max to avoid plotting problems
    end

    paths = [p.path_bandwidths]
    if p.plots_article push!(paths,p.path_plots_article) end
    for path in paths
        CairoMakie.save(string(path,name,".pdf"),fig)
    end
end

# w in meV, result in degrees
α2θ(α,w,p) = (180/π)*2*asin(w*1e-3*ev_to_hartree/(2*norm_K_cart(p.a)*p.vF*α))

function energy_center_zero(σs,p)
    l = fermi_label(p)
    σs2 = deepcopy(σs)
    for g=1:length(σs)
        σs2[g] = σs[g] .- σs[g][1,l]
    end
    σs2
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

########################## Plot Hamiltonians an other matrices

function save_H(H,p,title="H")
    rhm = heatmap(real.(H))
    ihm = heatmap(imag.(H))
    ahm = heatmap(abs.(H))
    pl = plot(rhm,ihm,ahm,size=(1000,700))
    savefig(pl,string(p.root_path,title,".png"))
end

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
    # CairoMakie.text!(Knames_tex, position=filt_pos,textsize=30)
    for i=1:length(filtered_names)
        CairoMakie.text!(Knames_tex[i], position=filt_pos[i],textsize=30)
    end

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
        save(path*"path_bands_diagram.pdf",f)
    end
    px("Momentum path in dual space plotted")
end

# From the numbers of the band diagram, produces a plot of it
function plot_band_diagram(σs,θs,name,p;K_relative=[0.0,0.0],shifts=zeros(100),energy_center=0,zero_central_energies=false,post_name="",colors=fill(:black,100)) # zero_central_energies : if true, the central energies will be at 0 artificially
    # Prepares the lists to plot
    n = length(p.Klist)
    res = p.resolution_bands
    n_path_points = res*n
    center = zero_central_energies ? 0 : energy_center
    ylims = center-p.energy_amplitude,center+p.energy_amplitude
    lengths_paths = [norm(k_red2cart(p.Klist[mod1(i+1,n)]-K_relative,p) .- k_red2cart(p.Klist[i]-K_relative,p)) for i=1:n]
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

    end_x = x_list[end]+dx_list[end] # fictitious x point, to loop the path
    CairoMakie.limits!(ax, x_list[1], end_x, ylims[1], ylims[2]) # x1, x2, y1, y2
    ax.yticks = (ylims[1] : 50 : ylims[2])

    # pl = Plots.plot(size=(1000,1100),ylims=ylims,legend=false) #:topright)
    σss = zero_central_energies ? energy_center_zero(σs,p) : σs
    n_l = length(σss[1][1,:])
    for g=1:length(σss)
        for l=1:n_l
            s = shifts[g]
            points = [CairoMakie.Point2f0(x_list[i],(σss[g][i,l] + s)*coef_plot_meV(θs[g],p)) for i=1:n_path_points]
            points = vcat(points,[CairoMakie.Point2f0(end_x,(σss[g][1,l] + s)*coef_plot_meV(θs[g],p))])
            CairoMakie.lines!(points,color=colors[g],linewidth=0.5)
        end
    end

    # Vertical lines and ticks
    list_x_vert_labels = vcat([x_list[(i-1)*res+1] for i=1:n],[x_list[end]+dx_list[end]])
    m = length(list_x_vert_labels)
    labels = [LaTeXString(string("\$",p.Klist_names[mod1(i,n)],"\$")) for i=1:n+1]

    pos = [CairoMakie.Point2f0(list_x_vert_labels[i] + (i==m ? -10 : 0),ylims[1]-abs(ylims[1])*0.0) for i=1:m]
    CairoMakie.text!(labels, position=pos,textsize=20,font ="Arial bold")
    ax.xticks = (list_x_vert_labels,["" for i=1:m])

    # Saves
    s = string(name,"000000000")
    title = s[1:min(6,length(s))]
    title = ""
    path_local = string(p.root_path,p.folder_plots_bands,"/")
    create_dir(path_local)
    paths_plots = [path_local]

    if p.plots_article push!(paths_plots,p.path_plots_article) end
    for path in paths_plots
        ext = path == path_local ? "png" : "pdf"
        path_plot = path*"band_struct"*title*"_"*post_name*"."*ext
        CairoMakie.save(path_plot,f)
    end
end

# Build T from a value of w
function BM(w,effV,p)
    effV.wAA = w*ev_to_hartree
    T = T_BM_four(effV.wAA,effV.wAA,effV)
    build_offdiag_V(T,p)
end

# Computes the band diagram for one value of θ
## parity_band : if true, gives the bands of H(-k) instead of H(k)
# kind ∈ ["new",110,126] is the operator we want. If "new", it takes cst_op as the part of the operator which does not depend on k, and takes Kf_new as the part which depends on k. If kind≂̸"new" it takes kind as a value in meV and builds the corresponding BM Hamiltonian
# Kf_new is the kinetic part of the new effective operator, Kf_pure is the one of the BM operator
function bands(kind,θ,p,EffV;parity_band=false,cst_op=[],Kf_new=[],Kf_pure=[]) 
    θrad = (π/180)*θ
    εθ = 2*sin.(θrad/2)
    cθ = cos(θrad/2)
    # p.coef_energies_plot = hartree_to_ev*1e3*εθ

    px("Computes band diagram, (N,d,θ)=(",p.N,",",EffV.interlayer_distance,",",θ,")")
    V = 0
    if kind=="new"
        V = cst_op
    else
        V = BM(kind/1e3,EffV,p)
    end
    V *= 1/εθ
    kin = kind=="new" ? (k -> Kf_new(k,cθ,εθ)) : Kf_pure
    P_kin = parity_band ? (k -> kin(-k)) : kin
    σ = spectrum_on_a_path(V,P_kin,p.Klist,p;print_progress=true)
    σ
end

# Creates and plots the band diagrams for some values of θ
# which ∈ ["new',"bm","all"]
function create_band_diagrams(θ_bm,θ_new,which,p,bands_fun)
    if which in ["new","all"]
        σ_new = bands_fun("new",θ_new)
        nmid = fermi_label(p)
        moy = (σ_new[1,nmid] + σ_new[1,nmid+1])/2
        shifts = [0,-moy]
        px("Shift in energy for the diagram: ",moy*coef_plot_meV(θ_new,p))
        plot_band_diagram([σ_new],[θ_new],"",p;post_name="eff_min_mw",colors=[:red],zero_central_energies=true,energy_center=floor(moy*coef_plot_meV(θ_new,p)))
    end
    if which in ["bm","all"]
        σ_bm = bands_fun(110,θ_bm)
        plot_band_diagram([σ_bm],[θ_bm],"",p;post_name="bm_min_bw",colors=[:black])
    end
end

# To create a figure with bandwidths, band gaps and Fermi velocities against θ
# alleviate ∈ ["new","110","126"], to compute only one of the band diagrams, for tests phase
function compute_bandwidths_and_velocity(θs,p,bands_partial;alleviate="all",true_bw=true)
    bw = [copy(zeros(length(θs))) for i=1:3]; bg = deepcopy(bw); fv = deepcopy(bw)
    for i=1:length(θs)
        θ = θs[i]
        if alleviate == "all"
            σ_new = bands_partial("new",θ)
            σ_bm_110 = bands_partial(110,θ)
            σ_bm_126 = bands_partial(126,θ)
        elseif alleviate == "new"
            σ_new = bands_partial("new",θ)
            σ_bm_110 = σ_new
            σ_bm_126 = σ_new
        elseif alleviate == 110
            σ_bm_110 = bands_partial(110,θ)
            σ_new = σ_bm_110
            σ_bm_126 = σ_bm_110
        elseif alleviate == 126
            σ_bm_126 = bands_partial(126,θ)
            σ_new = σ_bm_126
            σ_bm_110 = σ_bm_126
        end

        σs = [σ_new,σ_bm_110,σ_bm_126]
        for j=1:3
            (bw[j][i],bg[j][i],fv[j][i]) = bandwidth_bandgap_fermivelocity(σs[j],p)
        end

        px("θ = ",θ)
        px("Bandwidths ",bw[1][i]*coef_plot_meV(θ,p)," ",bw[2][i]*coef_plot_meV(θ,p)," ",bw[3][i]*coef_plot_meV(θ,p)," meV")
        px("Band gaps ", bg[1][i]*coef_plot_meV(θ,p)," ",bg[2][i]*coef_plot_meV(θ,p)," ",bg[3][i]*coef_plot_meV(θ,p)," meV")
        px("Fermi velocities ",fv[1][i]," ",fv[2][i]," ",fv[3][i])
    end
    name_doc = "exactV_and_all_differential_operators"
    plot_bandwidths_bandgaps_fermivelocities(θs,bw,bg,fv,p;def_ticks=true_bw,name=name_doc)
end

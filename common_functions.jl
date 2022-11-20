# using Plots#, LaTeXStrings
using DelimitedFiles, FFTW
px = println

import Base.+  
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...)  

####################################### Miscellaneous low level stuff

hartree_to_ev = 27.2114
ev_to_hartree = 1/hartree_to_ev
fill1d(x,n) = [deepcopy(x) for i=1:n]
fill2d(x,n) = [deepcopy(x) for i=1:n, j=1:n]
axis2grid(ax) = [(ax[i],ax[j]) for i=1:length(ax), j=1:length(ax)]
rotM(θ) = [cos(θ) -sin(θ);sin(θ) cos(θ)]
polar(x) = abs(x),atan(imag(x),real(x))
norm_K_cart(a) = 4*π/(3*a) # norm of Dirac's momentum
cyclic_conv(a,b,Vol) = myfft(myifft(a,Vol).*myifft(b,Vol),Vol)/sqrt(Vol) # convolution of Fourier vectors

# Creates a dictionary which inverts an array, for instance [5,4,1] will give Dict(5 => 1, 2 => 4, 3 => 1)
inverse_dict_from_array(a) = Dict(value => key for (key, value) in Dict(zip((1:length(a)), a)))

M2d_2_M3d(M) = [M           zeros(2,1);
                zeros(1,2)          1 ]

function myfloat2int(x;warning=true,name="") # float to int
    y = floor(Int,x+0.5)
    if abs(x-y)>1e-5 && warning
        px("NOT AN INT in ",name," ",x," ",y)
    end
    y
end

function distance(f,g) # relative distance between f and g, whatever they are
    nor = norm(f)
    if abs(nor) < 1e-15
        px("Division by zero in distance")
        return 0
    end
    norm(f.-g)/nor
end

function antiherm(M) # antihermitian part of M
    n = size(M,1)
    @assert size(M) == (n,n)
    s = sum(abs.(M))
    if s<1e-10
        return 0
    end
    sum(abs.((M'.-M)/2))/s
end

# if the directory path doesn't exist, it creates it, otherwise does nothing
create_dir(path) = if !isdir(path) mkdir(path) end

####################################### Fourier transforms

# If a_i = f(x_i) are the true values of function f in direct space, then myfft(f) gives the true Fourier coefficients
# where (𝔽f)_m := 1/sqrt(|Ω|) ∫_Ω f(x) e^{-ima^*x} dx are those coefficients, actually myfft(a)[m] = (𝔽f)_{m-1}
myfft(f,Vol) = FFTW.fft(f)*sqrt(Vol)/length(f)
myifft(f,Vol) = FFTW.ifft(f)*length(f)/sqrt(Vol)

####################################### Coordinates changes

cart2red_mat(M,p) = (p.lattice_2d')*M*inv(p.lattice_2d') # matrix of cartesian Fourier coords to reduced Fourier coords

function a_star_from_a(a1,a2) # vector in cartesian direct space to cartesian Fourier space
    lat = 2π*inv([a1 a2]')
    a1_star = lat[1:2,1]
    a2_star = lat[1:2,2]
    a1_star,a2_star
end

k_red2cart(k,p) = k[1]*p.a1_star + k[2]*p.a2_star # vector in reduced Fourier space to cartesian Fourier space
function cart2red_four(a,b,p) # G = ma^* to m, vector in cartesian Fourier space to reduced Fourier space
    S = cart2red_mat([a b],p)
    (S[1:2,1],S[1:2,2])
end


# (function k_inv_1d) from [k,l] which are momentums in reduced coordinates, ∈ ℤ^2, gives the label [ki,kl] so that C^D_{ki,li} (stored in numerics) = C_{k,l} (true coefficient of computations)
k_inv_1d(k,N) = Int(mod(k,N))+1 # from k in reduced Fourier coordinate to ki such that f^D[ki] = f_k, where f_k = (1/sqrt |Ω|) int e^{-ikx} f(x) dx and f^D is the array storing the coefficients f_k, k = fftfreq[ki] so k_inv_1d inverts fftfreq
k_inv(k,l,p) = (k_inv_1d(k,p.N),k_inv_1d(l,p.N))
kz_inv(k,p) = k_inv_1d(k,p.Nz)
k_inv_v(K,p) = (k_inv_1d(K[1],p.N),k_inv_1d(K[2],p.N))

####################################### Handles graphene cell structure

# Initialization of graphene cell structures
function init_cell_vectors(p;moire=true) # needs a. a_{i,M} = J a_i
    # Cell vectors
    a1_unit = [1/2;-sqrt(3)/2]
    a2_unit = [1/2; sqrt(3)/2]

    p.cell_area = sqrt(3)*0.5*p.a^2
    p.sqi = 1/sqrt(p.cell_area)
    p.a1,p.a2 = p.a.*(a1_unit,a2_unit)

    # If moiré, cell vectors are rotated
    if moire
        p.a1_micro,p.a2_micro = p.a1,p.a2 # keeps the information of microscopic graphene structure
        p.a1_star_micro,p.a2_star_micro = a_star_from_a(p.a1,p.a2)
        J = rotM(π/2)
        p.a1 = J*p.a1
        p.a2 = J*p.a2
    end

    # Builds dual cell vectors
    p.a1_star,p.a2_star = a_star_from_a(p.a1,p.a2)

    # Defines qj's TBG momentums
    if moire
        p.q1_red = -[1/3,1/3]
        p.q2_red = [2/3,-1/3]
        p.q3_red = [-1/3,2/3]
        p.q1 = -(1/3)*(p.a1_star.+p.a2_star)
        p.q2 = rotM(2π/3)*p.q1
        p.q3 = rotM(2π/3)*p.q2
    end

    # Builds lattice
    p.lattice_2d = [p.a1 p.a2]

    # Builds some Fourier operations
    p.R_four_2d = matrix_rot_red(-2π/3,p)
    p.M_four_2d = myfloat2int.(cart2red_mat([1 0;0 -1],p);name="M") # mirror symmetry, M_four_2d = [0 1;1 0]

    if p.dim == 3
        p.lattice_3d = [p.lattice_2d    zeros(2,1);
                        zeros(1,2)             p.L]
    end

    p.graphene_lattice_orientation = distance(p.a1,rotM(2π/3)*p.a2)<1e-5 || distance(p.a1,rotM(-2π/3)*p.a2)<1e-5 # true if the angle between a1 and a2 is 2π/3, false if it's π/3
    p.K_red = p.graphene_lattice_orientation ? -[1/3,1/3] : [1/3;-1/3] # Dirac point in reduced Fourier space
end

# Runs after init_cell_vectors, and one knows the discretization resolution. Initializes other quantities
function init_cell_infinitesimals(p;moire=true) # needs a, N, Nz ; only micro quantities !
    p.k_axis = myfloat2int.(fftfreq(p.N)*p.N;name="k_axis")
    p.k_grid = axis2grid(p.k_axis)
    p.kz_axis = myfloat2int.(fftfreq(p.Nz)*p.Nz;name="kz_axis")
    p.N2d = p.N^2
    p.N3d = p.N2d*p.Nz
    p.dS = p.cell_area/p.N^2 # surface element
    p.Vol = p.cell_area*p.L

    p.dx = p.a/p.N
    p.x_axis_cart = (0:p.N-1)*p.dx
    p.dz = p.L/p.Nz
    p.dv = p.Vol/(p.N^2*p.Nz)
end

####################################### Scalar products, integrations

function scaprod(ϕ,ψ,p,four=true) # scalar products
    d = length(size(ϕ))
    @assert d==length(size(ψ))
    if four
        return ϕ⋅ψ
    else
        dVol = d==1 ? p.dx : d==2 ? dS : p.dv
        return dVol*ϕ⋅ψ
    end
end
norm2(ϕ,p,four=true) = real(scaprod(ϕ,ϕ,p,four))
norms(ϕ,p,four=true) = sqrt(norm2(ϕ,p,four))

intZ(f,p) = p.dz*[sum(f[x,y,:]) for x=1:size(f,1), y=1:size(f,2)] # partial integration over XY
intXY(f,p) = p.dS*[sum(f[:,:,z]) for z=1:size(f,3)] # partial integration over Z
average_over_xy(f_dir,p) = [sum(f_dir[:,:,z]) for z=1:p.Nz]/p.N^2 # partial integration over Z

function Kinetic(u_four,p) # kinetic energy of u
    (∂1u,∂2u,∂3u) = ∇(u_four,p)
    norm2_3d(∂1u,p)+norm2_3d(∂2u,p)+norm2_3d(∂3u,p)
end

####################################### 3d functions

function substract_by_far_value(v_dir,p)
    # x,z = floor(Int,p.N/2),floor(Int,p.Nz/2)
    # m = v_dir[x,x,z]
    vz = average_over_xy(v_dir,p)
    Ns_mid = p.Nz/2; Ns1 = Ns_mid - p.Nz/10; Ns2 = Ns_mid + p.Nz/10
    Ns_mid,Ns1,Ns2 = Int.(floor.((Ns_mid,Ns1,Ns2)))
    sub = vz[Ns1:Ns2]
    m = sum(sub)/length(sub)
    v_dir .-= m
end

################## Computation of effective potentials and BM parameters

# Builds the Fourier coefficients 
# C_m = ∑_M conj(hat(g))_{m,M} hat(f)_{m,M} e^{i η d q_M 2π/L}, η ∈ {-1,1}
function build_Cm(g,f,p;η=1) 
    expo = [cis(2*η*p.interlayer_distance*kz*π/p.L) for kz in p.kz_axis]
    C = sqrt(p.cell_area)*[sum(conj.(g[m,n,:]).*f[m,n,:].*expo) for m=1:p.N, n=1:p.N]
    D = η==-1 ? C : parity_four(C,p)
    return D
end

function get_wAA_wC_from_fun(v_dir,p) # computes Bistritzer-MacDonald's wAA parameter
    u1v_dir = v_dir.*p.u1_dir
    u1_f = myfft(p.u1_dir,p.Vol)
    u1v_f = myfft(u1v_dir,p.Vol)
    C_Vu1_u1 = build_Cm(u1v_f,u1_f,p)
    wAA,wC = Tuple(real.([C_Vu1_u1[1,1],C_Vu1_u1[end,2]])).*p.sqi
    (wAA,wC)
end

function get_wAA_wC_from_monolayer(v_dir,p,vint_dir=-1) # Displays wAA in meV and wC the following dominant coefficient of the Fourier expansion
    px("##### wAA from monolayer functions")
    (wAA,wC) = get_wAA_wC_from_fun(v_dir,p)
    c = 1e3*hartree_to_ev
    px("wAA_v = ",c*wAA," meV ; wC_v = ",c*wC," meV")
    if vint_dir!=-1
        (wAA_vint,wC_vint) = get_wAA_wC_from_fun(vint_dir,p)
        wAA += wAA_vint; wC += wC_vint
        px("wAA_vint = ",c*wAA_vint," meV ; wC_vint = ",c*wC_vint," meV")
        px("wAA = ",c*wAA," meV ; wC = ",c*wC," meV")
    end
    px("#####")
    (wAA,wC)
end

####################################### Operations on functions in Fourier space

# V is M×M, frequency-sorted, and we want to get the N×N subset of values of V, where N<M. Cutoff of the Fourier functions (C_n)_{-M ≤ n ≤ M} to (C_n)_{-N ≤ n ≤ N} where N ≤ M (exact bounds would need to be adjusted)
function reduce_N_matrix(V,N) 
    M = size(V,1)
    cf = typeof(V[1,1])
    A = zeros(cf,N,N)
    @assert length(size(V))==2
    @assert N ≤ M
    kN = myfloat2int.(fftfreq(N)*N;name="kN")
    kM = myfloat2int.(fftfreq(M)*M;name="kM")
    for i=1:N, j=1:N
        kiN = kN[i]; kjN = kN[j]
        IM = k_inv_1d(kiN,M); JM = k_inv_1d(kjN,M)
        A[i,j] = V[IM,JM]
    end
    A
end

# translation u(x) := u(x - ya) = ∑_m u_m e^{ima^*(x-ya)}/sqrt(Vol) has Fourier coefs u_m e^{-i2π m⋅y}
function translation_interpolation(u_f,y_red,p) # y_red in reduced direct space, u_f in Fourier
    dim = length(size(u_f))
    if dim==3
        return [u_f[i,j,l]*cis(-2π*y_red⋅[p.k_axis[i],p.k_axis[j]]) for i=1:p.N, j=1:p.N, l=1:p.Nz]
    else
        return [u_f[i,j]*cis(-2π*y_red⋅[p.k_axis[i],p.k_axis[j]]) for i=1:p.N, j=1:p.N]
    end
end

matrix_rot_red(θ,p) = myfloat2int.(cart2red_mat(rotM(θ),p);name="mat_rot") # rotation operator in reduced Fourier space

# Gradient
# ∇f = i ∑_{m,m_z} hat(f)_m [ma^*[1];ma^*[2];m_z (2π/L)]
function ∇(f_four,p) # returns (∂1 f,∂2 f, ∂3 f) in Fourier space
    g1 = similar(f_four); g2 = similar(f_four); g3 = similar(f_four)
    for m=1:p.N, n=1:p.N
        (m0,n0) = p.k_grid[m,n]
        k = m0*p.a1_star .+ n0*p.a2_star
        c = f_four[m,n,:]
        g1[m,n,:] = c.*k[1]
        g2[m,n,:] = c.*k[2]
        g3[m,n,:] = c.*(2π/p.L).*p.kz_axis
    end
    im.*(g1,g2,g3)
end

R_four(a,p) = apply_map_four(X ->  p.R_four_2d*X,a,p) # rotation of 2π/3, in Fourier space
M_four(a,p) = apply_map_four(X ->  p.M_four_2d*X,a,p) # Mu(x,y) := u(x,-y), mirror
# J_four(a,p) = apply_map_four(X -> -[1 -2;2 -1]*X,a,p) # rotation of -π/2, in Fourier space, with scaling of sqrt(3)
parity_four(a,p) = apply_map_four(X -> -X,a,p)
conj_four(a,p) = apply_map_four(X -> -X,conj.(a),p) # if f is in direct space, and g(x) := conj.(f(x)), then hat(g)_m = conj(hat(f))_{-m}
σ1_four(a,p) = apply_map_four(X -> [0 1;1 0]*X,a,p)

function apply_map_four(L,u,p) # gives C_m = u_{Lm}, ie applies L in Fourier reduced space. u is in Fourier
    dim = length(size(u)); tu = typeof(u[1])
    a = dim==2 ? zeros(tu,p.N,p.N) : zeros(tu,p.N,p.N,p.Nz)
    for K=1:p.N, P=1:p.N
        k0 = p.k_axis[K]; p0 = p.k_axis[P]
        c = L([k0,p0])
        k1 = c[1]; p1 = c[2]
        (k2,p2) = k_inv(k1,p1,p)
        if dim==2
            a[K,P] = u[k2,p2]
        else 
            a[K,P,:] = u[k2,p2,:]
        end
    end
    a
end

function apply_map_four_back(L,u,p) # gives C_m such that C_{Lm} = u_m
    a = zeros(typeof(u[1]),p.N,p.N)
    for K=1:p.N, P=1:p.N
        k0 = p.k_axis[K]
        p0 = p.k_axis[P]
        c = L([k0,p0]); k1 = c[1]; p1 = c[2]
        (k2,p2) = k_inv(k1,p1,p)
        a[k2,p2] = u[K,P]
    end
    a
end

J_four_back(a,p) = apply_map_four_back(X -> -[1 -2;2 -1]*X,a,p) # rotation of -π/2, in Fourier space, with scaling of sqrt(3)

####################################### Coordinates changes in periodic direct space
# Example
# xi = [1 2 3 4]
#  x = [0 1 2 3]
xi2x(x,p) = mod(x-1,p.N)
x2xi(x,p) = mod1(x+1,p.N)

# Xi = xi + n*N in natural coordinates corresponding to xi
# Xpi = (xi,n)
# X = Xi-1
# Xred = x_red[xi] + n*p.L
# Xp = (x,n)
Xpi2Xp(Xpi,p) = (xi2x(Xpi[1],p),Xpi[2])
Xp2Xpi(X,p) = (x2xi(X[1],p),n)
Xpi2Xred(X,p) = p.x_axis_red[X[1]] + p.La0*X[2] # NECESSITATES THAT L1 == L2 !!!

Xi2Xpi(xi,p) = (mod1(xi,p.N),fld1(xi,p.N))
X2Xi(X) = X+1
Xi2X(Xi) = Xi-1
X2Xpi(X,p) = Xi2Xpi(X2Xi(X),p)

####################################### Operations on functions in direct space

# Applies coordinates transformation M
function apply_coordinates_operation_direct(M,p) # (Op f)(x) = f(Mx), 2d or 3d
    function f(ϕ,p)
        ψ = similar(ϕ)
        dim = length(size(ϕ))
        for xi=1:p.N, yi=1:p.N
            X = Int.(M([xi2x(xi,p);xi2x(yi,p)]))
            Xi = x2xi(X[1],p); Yi = x2xi(X[2],p)
            if dim == 3
                ψ[xi,yi,:] = ϕ[Xi,Yi,:]
            else
                ψ[xi,yi] = ϕ[Xi,Yi]
            end
        end
        ψ
    end
    f
end

function apply_coordinates_operation_direct_on_z(M,p)
    function f(ϕ,p)
        ψ = similar(ϕ)
        dim = length(size(ϕ))
        for zi=1:p.N
            z = Int.(M(xi2x(zi,p)))
            zi = x2xi(z,p)
            ψ[:,:,zi] = ϕ[:,:,zi]
        end
        ψ
    end
    f
end

# R(ϕ,p)             = apply_coordinates_operation_direct(X -> p.R_four_2d*X,p)(ϕ,p)
translation(ϕ,v,p) = apply_coordinates_operation_direct(X -> X.-v,p)(ϕ,p)
parity_x(ϕ,p)      = apply_coordinates_operation_direct(X -> -X,p)(ϕ,p)

translation2d(u,a,p) = [u[mod1(x-a[1],p.N),mod1(y-a[2],p.N)] for x=1:p.N, y=1:p.N]

function parity_z(u,p)
    dim = length(size(u))
    if dim == 3
        return [u[x,y,mod1(2-z,p.Nz)] for x=1:p.N, y=1:p.N, z=1:p.Nz]
    else
        return [u[mod1(2-z,p.Nz)] for z=1:p.Nz]
    end
end

# Applies coordinates transformation M and does a Bloch transform at the same time
# (Op B f)(x) = (Op ∘ exp)(x) * (Op ∘ f)(x) = exp(Mx) * f(Mx)
function apply_Op_B(M,k,p) 
    function f(u,k,p)
        ψ = similar(u)
        for xi=1:p.N, yi=1:p.N
            # Rotation
            RX = Int.(M([xi2x(xi,p);xi2x(yi,p)]))
            Xpi = X2Xpi(RX[1],p); Ypi = X2Xpi(RX[2],p)

            # Phasis
            X = Xpi2Xred(Xpi,p); Y = Xpi2Xred(Ypi,p)
            φ = cis(2π*k⋅[X;Y])
            ψ[xi,yi,:] = φ*u[Xpi[1],Ypi[1],:]
        end
        ψ
    end
    f
end

# Bloch transform and Rotation, RBu = Rexp(...) * Ru. We obtain (RBu)(x,y) for (x,y) ∈ x_grid_red
RB(u,k,p) = apply_Op_B(X -> p.R_four_2d*X,k,p)(u,k,p)
PB(u,k,p) = apply_Op_B(X -> -X,k,p)(u,k,p)

####################################### Test functions

function test_scaprod_fft_commutation(p) # tests commutation between FFT and scalar product operations
    ϕ = randn(p.N,p.N,p.N)
    ψ = randn(p.N,p.N,p.N)
    c = scaprod(ϕ,ψ,p,false) - scaprod(myfft(ϕ,p.Vol),myfft(ψ,p.Vol),p)
    px("Test scalar product ",c)
end

function test_k_inv() # tests kz inversion
    N = 10
    k_axis = fftfreq(N)*N
    c = true
    for ki=1:N
        k = k_axis[ki]
        if k_axis[k_inv_1d(k,N)] != k
            c = false
        end
    end
    px("Test k inv: ",c ? "good" : "problem")
end

test_hermitianity(M,name="") = px(string("Test Hermitianity ",name," : "),antiherm(M)) # tests M^*=M

function test_x_parity(u,p;name="") # Tests u(-x) = u(x) (or u(-x,z) = u(x,z)), where u is in direct space
    c = sum(abs.(parity_x(u,p) .- u))/sum(abs.(u))
    px("Test ",name,"(-x) = ",name,"(x) : ",c)
end

# tests u(-z) = ε u(z)
test_z_parity(u,ε,p;name="function") = px("Test ",name,"(-z) = ",ε==-1 ? "-" : "",name,"(z) : ",norm2(u.- ε*parity_z(u,p),p)/norm2(u,p))

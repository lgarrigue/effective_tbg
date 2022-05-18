# using Plots#, LaTeXStrings
px = println

import Base.+  
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...)  

fill2d(x,n) = [copy(x) for i=1:n, j=1:n]
axis2grid(ax) = [(ax[i],ax[j]) for i=1:length(ax), j=1:length(ax)]
rotM(θ) = [cos(θ) -sin(θ);sin(θ) cos(θ)]
scale_fun2d(f,λ) = (x,y) -> f(λ*x,λ*y) # x -> f(λx)
scale_fun3d(f,λ) = (x,y,z) -> f(λ*x,λ*y,λ*z)

# Creates a dictionary which inverts an array, for instance [5,4,1] will give Dict(5 => 1, 2 => 4, 3 => 1)
inverse_dict_from_array(a) = Dict(value => key for (key, value) in Dict(zip((1:length(a)), a)))

M2d_2_M3d(M) = [M           zeros(2,1);
		zeros(1,2)          1 ]

hartree_to_ev = 27.2114
ev_to_hartree = 1/hartree_to_ev

function inverse_dict_from_2d_array(a)
	direct = Dict(zip((1:length(a)), a))
	Ci = CartesianIndices(a)
	Dict(value => (Ci[key][1],Ci[key][2]) for (key, value) in direct)
end



function myfloat2int(x;warning=true)
	y = floor(Int,x+0.5)
	if abs(x-y)>1e-5 && warning
		px("NOT AN INT ",x," ",y)
	end
	y
end

cart2red_mat(M,p) = (p.lattice_2d')*M*inv(p.lattice_2d')

function cart2red_four(a,b,p) # G = ma^* to m 
	S = cart2red_mat([a b],p)
	(S[1:2,1],S[1:2,2])
end

matrix_rot_red(θ,p) = myfloat2int.(cart2red_mat(rotM(θ),p))

function a_star_from_a(a1,a2)
	lattice_rec = 2π*inv([a1 a2]')
	a1_star = lattice_rec[1:2,1]
	a2_star = lattice_rec[1:2,2]
	a1_star,a2_star
end

length_a_to_length_a_star(a) = 4π/(a*sqrt(3))

function init_cell_vectors(p;moire=true) # needs a. a_{i,M} = (1/2) J a_i
	a1_unit = [1/2;-sqrt(3)/2]
	a2_unit = [1/2; sqrt(3)/2]

	a = moire ? p.a_micro : p.a
	p.a1,p.a2 = a.*(a1_unit,a2_unit) # not used

	if moire
		p.a1_micro,p.a2_micro = p.a1,p.a2
		p.a1_star_micro,p.a2_star_micro = a_star_from_a(p.a1,p.a2)
		J = rotM(π/2)
		p.a1 = (1/2)*J*p.a1
		p.a2 = (1/2)*J*p.a2
		p.a = norm(p.a1)
		p.cell_area = sqrt(3)*0.5*p.a^2
		p.cell_area_micro = sqrt(3)*0.5*p.a_micro^2
	else
		p.cell_area = sqrt(3)*0.5*p.a^2
	end

	p.a1_star,p.a2_star = a_star_from_a(p.a1,p.a2)

	if moire
		p.q1 = -(1/3)*(p.a1_star.+p.a2_star)
		p.q2 = rotM(2π/3)*p.q1
		p.q3 = rotM(2π/3)*p.q2
	end

	# a1s_unit = [sqrt(3)/2;-1/2]
	# a2s_unit = [sqrt(3)/2; 1/2]
	# a1_star,a2_star = (4π/(p.a*sqrt(3))).*(a1s_unit,a2s_unit)

	p.lattice_2d = [p.a1 p.a2]
	p.R_four_2d = matrix_rot_red(-2π/3,p)
	p.M_four_2d = myfloat2int.(cart2red_mat([1 0;0 -1],p)) # mirror symmetry

	if p.dim == 3
		p.lattice_3d = [p.lattice_2d    zeros(2,1);
				zeros(1,2)             p.L]
	end

	p.lattice_type_2πS3 = distance(p.a1,rotM(2π/3)*p.a2)<1e-5 || distance(p.a1,rotM(-2π/3)*p.a2)<1e-5 
	px("type ",p.lattice_type_2πS3)
	p.K_red = p.lattice_type_2πS3 ? [1/3,1/3] : [-1/3;1/3]
end

function update_a(a1_star,a2_star,p)
	p.a1_star = a1_star
	p.a2_star = a2_star
	p.a1,p.a2 = a_star_from_a(p.a1_star,p.a2_star)
	p.a = norm(p.a1)
	p.cell_area = sqrt(3)*0.5*p.a^2
	p.lattice_2d = [p.a1 p.a2]
	p.lattice_type_2πS3 = distance(p.a1,rotM(2π/3)*p.a2)<1e-5 || distance(p.a1,rotM(-2π/3)*p.a2)<1e-5 
	p.K_red = p.lattice_type_2πS3 ? [1/3,1/3] : [-1/3;1/3]
end

# Runs after init_cell_vectors
function init_cell_infinitesimals(p;moire=true) # needs a, N, Nz ; only micro quantities !
	p.k_axis = Int.(fftfreq(p.N)*p.N)
	p.k_grid = axis2grid(p.k_axis)
	p.kz_axis = Int.(fftfreq(p.Nz)*p.Nz)
	p.N2d = p.N^2; p.N3d = p.N2d*p.Nz
	p.dS = p.cell_area/p.N^2
	if moire
		p.dS_micro = p.cell_area_micro/p.N^2
		p.Vol = p.cell_area_micro*p.L
	else
		p.dx = p.a/p.N
		p.x_axis_cart = (0:p.N-1)*p.dx
		p.Vol = p.cell_area*p.L
		p.dz = p.L/p.Nz
	end
	p.dv = p.Vol/(p.N^2*p.Nz)
end

function k_red2cart_list(k1,k2,p)
	k_cart = k1*p.a1_star+k2*p.a2_star
	(k_cart[1],k_cart[2])
end

k_red2cart(k,p) = k[1]*p.a1_star + k[2]*p.a2_star
k_ind2cart(mix,miy,p) = p.k_axis[mix]*p.a1_star + p.k_axis[miy]*p.a2_star # indices to cartesian vector G

fill2d(x,n) = [copy(x) for i=1:n, j=1:n]
fill1d(x,n) = [copy(x) for i=1:n]
init_vec(p) = fill2d(zeros(ComplexF64,p.N,p.N),4)

cyclic_conv(a,b,Vol) = myfft(myifft(a,Vol).*myifft(b,Vol),Vol)/sqrt(Vol)


# translation u(x) := u(x - ya) = ∑_m u_m e^{ima^*(x-ya)}/sqrt(Vol) has Fourier coefs u_m e^{-i2π m⋅y}
function translation_interpolation(u_f,y_red,p)
	dim = length(size(u_f))
	if dim==3
		return [u_f[i,j,l]*cis(-2π*y_red⋅[p.k_axis[i],p.k_axis[j]]) for i=1:p.N, j=1:p.N, l=1:p.Nz]
	else
		return [u_f[i,j]*cis(-2π*y_red⋅[p.k_axis[i],p.k_axis[j]]) for i=1:p.N, j=1:p.N]
	end
end

function scaprod(ϕ,ψ,p,four=true)
	d = length(size(ϕ))
	@assert d==length(size(ψ))
	dVol = d==1 ? p.dx : d==2 ? dS : p.dv
	(four ? 1 : dVol)*ϕ⋅ψ
end

function test_scaprod_fft_commutation(p)
	ϕ = randn(p.N,p.N,p.N)
	ψ = randn(p.N,p.N,p.N)
	c = scaprod(ϕ,ψ,p,false) - scaprod(myfft(ϕ,p.Vol),myfft(ψ,p.Vol),p)
	px("Test scalar product ",c)
end

norm2(ϕ,p,four=true) = real(scaprod(ϕ,ϕ,p,four))
norms(ϕ,p,four=true) = sqrt(norm2(ϕ,p,four))

####################################### 3d functions

intZ(f,p) = p.dz*[sum(f[x,y,:]) for x=1:size(f,1), y=1:size(f,2)]
intXY(f,p) = p.dS*[sum(f[:,:,z]) for z=1:size(f,3)]

axis2grid_ar(ax) = [[ax[i],ax[j]] for i=1:length(ax), j=1:length(ax)]

# from [k,l] which are momentums in reduced coordinates, ∈ ℤ^2, gives the label [ki,kl] so that C^D_{ki,li} (stored in numerics) = C_{k,l} (true coefficient of computations)

k_inv_1d(k,N) = Int(mod(k,N))+1 # from k in reduced Fourier coordinate to ki such that f^D[ki] = f_k, where f_k = int e^{-ikx} f(x) dx and f^D is the array storing the coefficients f_k, k = fftfreq[ki] so k_inv_1d inverts fftfreq
k_inv(k,l,p) = (k_inv_1d(k,p.N),k_inv_1d(l,p.N))
kz_inv(k,p) = k_inv_1d(k,p.Nz)

function test_k_inv()
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

# tests whether u(-z) = ε u(z)

# ∇f = i ∑_{m,m_z} hat(f)_m [ma^*[1];ma^*[2];m_z (2π/L)]
function ∇(f_four,p) # returns (∂1 f,∂2 f, ∂3 f)
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

function Kinetic(u_four,p) # kinetic energy of u
	(∂1u,∂2u,∂3u) = ∇(u_four,p)
	norm2_3d(∂1u,p)+norm2_3d(∂2u,p)+norm2_3d(∂3u,p)
end

####################################### Operations on functions in Fourier space

R_four(a,p) = apply_map_four(X ->  p.R_four_2d*X,a,p) # rotation of 2π/3, in Fourier space
M_four(a,p) = apply_map_four(X ->  p.M_four_2d*X,a,p) # Mu(x,y) := u(x,-y)
# J_four(a,p) = apply_map_four(X -> -[1 -2;2 -1]*X,a,p) # rotation of -π/2, in Fourier space, with scaling of sqrt(3)
parity_four(a,p) = apply_map_four(X -> -X,a,p)
conj_four(a,p) = apply_map_four(X -> -X,conj.(a),p) # if f is in direct space, and g(x) := conj.(f(x)), hat(g)_m = conj(hat(f))_{-m}
σ1_four(a,p) = apply_map_four(X -> [0 1;1 0]*X,a,p)

function apply_map_four(L,u,p) # res_m = u_{Lm}
	dim = length(size(u)); tu = typeof(u[1])
	a = dim==2 ? zeros(tu,p.N,p.N) : zeros(tu,p.N,p.N,p.Nz)
	for K=1:p.N, P=1:p.N
		k0 = p.k_axis[K]; p0 = p.k_axis[P]
		c = L([k0,p0]); k1 = c[1]; p1 = c[2]
		(k2,p2) = k_inv(k1,p1,p)
		if dim==2
			a[K,P] = u[k2,p2]
		else 
			a[K,P,:] = u[k2,p2,:]
		end
	end
	a
end

function apply_map_four_back(L,u,p) # res_{Lm} = u_m
	a = zeros(typeof(u[1]),p.N,p.N)
	for K=1:p.N, P=1:p.N
		k0 = p.k_axis[K]; p0 = p.k_axis[P]
		c = L([k0,p0]); k1 = c[1]; p1 = c[2]
		(k2,p2) = k_inv(k1,p1,p)
		a[k2,p2] = u[K,P]
	end
	a
end

J_four_back(a,p) = apply_map_four_back(X -> -[1 -2;2 -1]*X,a,p) # rotation of -π/2, in Fourier space, with scaling of sqrt(3)

####################################### Operations on functions in direct space

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

R(ϕ,p)             = apply_coordinates_operation_direct(X -> p.R_four_2d*X,p)(ϕ,p)
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

# P(ϕ,p) = [ϕ[p.parity_axis[x],p.parity_axis[y]] for x=1:p.N, y=1:p.N] # Pf(x) = f(-x)

# Applies coordinates transformation M and does a Bloch transform
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

######################## Symmetry tests

function distance(f,g)
	nor = norm(f)
	if abs(nor) < 1e-15
		px("Division by zero in distance")
		return 0
	end
	norm(f.-g)/nor
end

function herm(M)
	n = size(M,1)
	@assert size(M) == (n,n)
	s = sum(abs.(M))
	if s<1e-10
		return 0
	end
	sum(abs.((M'.-M)/2))/s
end

test_hermitianity(M,name="") = px(string("Test Hermitianity ",name," : "),herm(M))

function test_x_parity(u,p;name="") # Tests u(-x) = u(x) (or u(-x,z) = u(x,z)), where u is in direct space
	c = sum(abs.(parity_x(u,p) .- u))/sum(abs.(u))
	px("Test ",name,"(-x) = ",name,"(x) : ",c)
end

test_z_parity(u,ε,p;name="function") = px("Test ",name,"(-z) = ",ε==-1 ? "-" : "",name,"(z) : ",norm2(u.- ε*parity_z(u,p),p)/norm2(u,p))

# Bloch transform and Rotation, RBu = Rexp(...) * Ru. We obtain (RBu)(x,y) for (x,y) ∈ x_grid_red
RB(u,k,p) = apply_Op_B(X -> p.R_four_2d*X,k,p)(u,k,p)
PB(u,k,p) = apply_Op_B(X -> -X,k,p)(u,k,p)


######################## Coordinates changes in periodic direct space
# Example
# xi = [1 2 3 4]
#  x = [0 1 2 3]
xi2x(x,p) = mod(x-1,p.N)
x2xi(x,p) = mod1(x+1,p.N)

######################## Coordinates changes in entire direct space
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

######################## Coordinates changes in periodic Fourier space
# ks = k_shifted, k_nat = k_natural
# Example for N=6
#    ki = [ 1  2  3  4  5  6]
# k_nat = [ 0  1  2 -3 -2 -1] from fftfreq(N)
#    ks = [-3 -2 -1  0  1  2] from fftshift(fftfreq(N))

# Example for N=5
#    ki = [ 1  2  3  4  5]
# k_nat = [ 0  1  2 -2 -1]
#    ks = [-2 -1  0  1  2]
ki2k_nat(ki,N) = fftfreq(N,N)[ki]

# from k ∈ ℤ to its representent in integer coordinates. For ex. N=6, -3 -> 4 and 2 -> 3
k_nat2ki(k,N) = Int(mod(k,N))+1

# from ki ∈ [1,...,N] (label coordinates) to its representent k ∈ [-N/2,...,N/2] in shifted coordinates
ki2ks(ki,N) = ki-1-Int(floor(N/2))

# from k ∈ ℤ to ki
ks2ki(k,N) = Int(mod(k-Int(floor(N/2)),N))+1

# from k ∈ ℤ (in shifted coordinates) to its representent k ∈ [-N/2,...,N/2] in shifted coordinates
ks2ks(k,N) = Int(mod(k-Int(floor(N/2)),N))-Int(floor(N/2))
k_nat2ks(k,N) = Int(mod(k,N))-Int(floor(N/2))
ks2k_nat(k,N) = ki2k_nat(ks2ki(k,N),N)
fun2d(f,k,N) = [f(k[1],N),f(k[2],N)]
ki2ki(k,N) = mod1(k,N)

# action of a matrix on a vector k, given in coordinates ki ∈ [1,...,N], returns coordinates in ki
function mat_on_ki(Q,ki,p) 
	k = fun2d(ki2k_nat,ki,p.N)
	fun2d(k_nat2ki,Q*k,p.N)
end

######################## Fourier transforms
# If a_i = f(x_i) are the true values of function f in direct space, then myfft(f) gives the true Fourier coefficients
# where (𝔽f)_m := 1/sqrt(|Ω|) ∫_Ω f(x) e^{-ima^*x} dx are those coefficients, actually myfft(a)[m] = (𝔽f)_{m-1}
myfft(f,Vol) = fft(f)*sqrt(Vol)/length(f)
myifft(f,Vol) = ifft(f)*length(f)/sqrt(Vol)

######################## Plot function 1d

function red_arr2fun_red_1d(ψ_four,vol) 
	N = length(ψ_four)
	k_axis = fftfreq(N)*N
	f(x) = 0
	for i=1:N
		g(x) = ψ_four[i] * cis(2π*k_axis[i]*x)/sqrt(vol)
		f = f + g
	end
	f
end

function eval_fun_to_plot_1d(ψ_four,res,vol)
	f = red_arr2fun_red_1d(ψ_four,vol)
	real.(f.((0:res-1)/res))
end

######################## Plot

function plot2d(f,p;axis=nothing,size=(100,200))
	heatmap(f,axis=axis,size=size)
end

function plot_spectrum(E)
	plo = plot(title="σ",xaxis=([], false),grid=false,ylabel="E")
	indent = 0
	n = length(E)
	for l=1:n
		indent += 0.3
		plot!(plo,[indent-0.1,indent+0.1],[E[l],E[l]])
		if l<length(E)
			if E[l+1]-E[l] > (E[n]-E[1])/40
				indent = 0
			end
		end
	end
	label = [string(i) for i=1:length(E)]
	plot!(plo,label=label,legend=:bottomright)
	plo
end

function plotVeff(V,f,p)
	htms = [plot2d(f.(V[i,j]),p) for i=1:4, j=1:4]
	plot(htms...,size=(1200,1300),layout=(4,4))
end

function plotVbm(Vbm,Vbm_dir,p,f=real)
	plbm = plotVeff(Vbm,f,p)
	plbm_dir = plotVeff(Vbm_dir,f,p)
	pl = plot(plbm,plbm_dir,layout=(2,1))
	savefig(pl,"Vbm.png")
end

function plot_H(H,f,b)
	A = fill2d(zeros(ComplexF64,b.N,b.N),4)
	for i=1:4, j=1:4, ck_lin=1:b.N2d
		(pix,piy) = lin_k2coord_ik(ck_lin,b)
		c1 = b.Li_tot[i,pix,piy]
		c2 = b.Li_tot[j,pix,piy]
		A[i,j][pix,piy] = H[c1,c2]
		# if i==4 && j==4
		# px(H[c1,c2])
		# end
	end
	pl = plotVeff(A,f,b)
	savefig(pl,"H.png")
end

######################## Misc

# if the directory path doesn't exist, it creates it, otherwise does nothing
create_dir(path) = if !isdir(path) mkdir(path) end

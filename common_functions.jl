using Plots#, LaTeXStrings
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

function inverse_dict_from_2d_array(a)
	direct = Dict(zip((1:length(a)), a))
	Ci = CartesianIndices(a)
	Dict(value => (Ci[key][1],Ci[key][2]) for (key, value) in direct)
end

function init_cell_infinitesimals(p) # needs a, N, Nz
	p.cell_area = sqrt(3)*0.5*p.a^2 
	p.dx = p.a/p.N
	p.dS = p.cell_area/p.N^2
	p.k_axis = Int.(fftfreq(p.N)*p.N)
	p.k_grid = axis2grid(p.k_axis)
	p.x_axis_cart = (0:p.N-1)*p.dx
	p.N2d = p.N^2; p.N3d = p.N2d*p.Nz

	# z-dim quantities
	p.dz = p.L/p.Nz
	p.dv = p.dS*p.dz
	p.kz_axis = Int.(fftfreq(p.Nz)*p.Nz)
	p.Vol = p.cell_area*p.L
end

function init_cell_vectors(p) # needs a
	a1_unit = [sqrt(3)/2; 1/2]
	a2_unit = [sqrt(3)/2;-1/2]
	a1s0 = [-1; 1/sqrt(3)]
	a2s0 = [ 1; 1/sqrt(3)]
	p.a1 = p.a*a1_unit
	p.a2 = p.a*a2_unit
	p.a1_star = (2π/p.a)*a1s0
	p.a2_star = (2π/p.a)*a2s0
	J = rotM(π/2)
	# p.a1_star = J*p.a1_star
	# p.a2_star = J*p.a2_star

	p.K_red = [-1/3;1/3]
end

fill2d(x,n) = [copy(x) for i=1:n, j=1:n]
fill1d(x,n) = [copy(x) for i=1:n]
init_vec(p) = fill2d(zeros(ComplexF64,p.N,p.N),4)

cyclic_conv(a,b) = prod(size(a))*fft(ifft(a).*ifft(b))

####################################### 2d functions

integral2d(ϕ,p,four=true) = p.dS*sum(ϕ)/(!four ? 1 : p.N2d)
sca2d(ϕ,ψ,p,four=true) = p.dS*ϕ⋅ψ/(!four ? 1 : p.N2d)
norm2_2d(ϕ,p,four=true) = real(sca2d(ϕ,ϕ,p,four))
norms2d(ϕ,p,four=true) = sqrt(norm2_2d(ϕ,p,four))

####################################### 3d functions

intZ(f,p) = p.dz*[sum(f[x,y,:]) for x=1:size(f,1), y=1:size(f,2)]
intXY(f,p) = p.dS*[sum(f[:,:,z]) for z=1:size(f,3)]

integral3d(ϕ,p,four=true) = p.dv*sum(ϕ)/(!four ? 1 : p.N3d)
sca3d(ϕ,ψ,p,four=true) = p.dv*ϕ⋅ψ/(!four ? 1 : p.N3d)
norm2_3d(ϕ,p,four=true) = real(sca3d(ϕ,ϕ,p,four))
norms3d(ϕ,p,four=true) = sqrt(norm2_3d(ϕ,p,four))

sca3d_four(f,g,p) = p.Vol*f⋅g
norm2_3d_four(f,p) = real(sca3d_four(f,f,p))
norms_3d_four(f,p) = sqrt(norm2_3d_four(f,p))

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
	px("Test k inv ",c)
end

# tests whether u(-z) = ε u(z)

function ∇(f_four,p) # returns (∂1 f,∂2 f, ∂3 f)
	g1 = similar(f_four); g2 = similar(f_four); g3 = similar(f_four)
	for m=1:p.N
		for n=1:p.N
			# Under cart-to-red change, ∇ calK^{-1} = calK^{-1} (K^*)^{-1} ∇
			(m0,n0) = p.k_grid[m,n]
			k = m0*p.a1_star .+ n0*p.a2_star
			# px("GRID ",p.k_grid[m,n]," ",a)
			g1[m,n,:] = im*2π*f_four[m,n,:]*k[1]
			g2[m,n,:] = im*2π*f_four[m,n,:]*k[2]
			g3[m,n,:] = im*2π*f_four[m,n,:] .* p.kz_axis
		end
	end
	(g1,g2,p.L*g3)
end

function Kinetic(u_four,p) # kinetic energy of u
	(∂1u,∂2u,∂3u) = ∇(u_four,p)
	norm2_3d(∂1u,p)+norm2_3d(∂2u,p)+norm2_3d(∂3u,p)
end

####################################### Operations on functions in Fourier space

R_four(a,p) = apply_map_four(X -> [0 -1;1 -1]*X,a,p) # rotation of 2π/3, in Fourier space
J_four(a,p) = apply_map_four(X -> -[1 -2;2 -1]*X,a,p) # rotation of -π/2, in Fourier space, with scaling of sqrt(3)
parity_four(a,p) = apply_map_four(X -> -X,a,p)
σ1_four(a,p) = apply_map_four(X -> [0 1;1 0]*X,a,p)

function apply_map_four(L,u,p) 
	a = similar(u)
	# a = zeros(ComplexF64,p.N,p.N)
	for K=1:p.N
		for P=1:p.N
			k0 = p.k_axis[K]; p0 = p.k_axis[P]
			c = L([k0,p0]); k1 = c[1]; p1 = c[2]
			(k2,p2) = k_inv(k1,p1,p)
			a[K,P] = u[k2,p2]
		end
	end
	a
end

####################################### Operations on functions in direct space

function apply_coordinates_operation_direct(M,p) # (Op f)(x) = f(Mx), 2d or 3d
	function f(ϕ,p)
		ψ = similar(ϕ)
		dim = length(size(ϕ))
		for xi=1:p.N
			for yi=1:p.N
				X = Int.(M([xi2x(xi,p);xi2x(yi,p)]))
				Xi = x2xi(X[1],p); Yi = x2xi(X[2],p)
				if dim == 3
					ψ[xi,yi,:] = ϕ[Xi,Yi,:]
				else
					ψ[xi,yi] = ϕ[Xi,Yi]
				end
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

R(ϕ,p)             = apply_coordinates_operation_direct(X -> p.mat_R*X,p)(ϕ,p)
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
		for xi=1:p.N
			for yi=1:p.N
				# Rotation
				RX = Int.(M([xi2x(xi,p);xi2x(yi,p)]))
				Xpi = X2Xpi(RX[1],p); Ypi = X2Xpi(RX[2],p)

				# Phasis
				X = Xpi2Xred(Xpi,p); Y = Xpi2Xred(Ypi,p)
				φ = cis(2π*k⋅[X;Y])
				ψ[xi,yi,:] = φ*u[Xpi[1],Ypi[1],:]
			end
		end
		ψ
	end
	f
end

######################## Symmetry tests

function test_hermitianity(M,name="")
	n = size(M,1)
	@assert size(M) == (n,n)
	s = sum(abs.(M))
	x = s < 1e-10 ? 0 : sum(abs.((M'.-M)/2))/s
	px(string("Test Hermitianity ",name," : "),x)
end

function test_x_parity(u,p;name="") # Tests u(-x) = u(x) (or u(-x,z) = u(x,z)), where u is in direct space
	c = sum(abs.(parity_x(u,p) .- u))/sum(abs.(u))
	px("Test ",name,"(-x) = ",name,"(x) : ",c)
end

test_z_parity(u,ε,p;name="function") = px("Test ",name,"(-z) = ",ε==-1 ? "-" : "",name,"(z) : ",norm2_3d(u.- ε*parity_z(u,p),p)/norm2_3d(u,p))

# Bloch transform and Rotation, RBu = Rexp(...) * Ru. We obtain (RBu)(x,y) for (x,y) ∈ x_grid_red
RB(u,k,p) = apply_Op_B(X -> p.mat_R*X,k,p)(u,k,p)
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

######################## Plot function 1d

function red_arr2fun_red_1d(ψ_four) 
	N = length(ψ_four)
	k_axis = fftfreq(N)*N
	f(x) = 0
	for i=1:N
		g(x) = ψ_four[i] * cis(2π*k_axis[i]*x)
		f = f + g
	end
	f
end

function eval_fun_to_plot_1d(ψ_four,res)
	f = red_arr2fun_red_1d(ψ_four) 
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
	for i=1:4
		for j=1:4
			for ck_lin=1:b.N2d
				(pix,piy) = lin_k2coord_ik(ck_lin,b)
				c1 = b.Li_tot[i,pix,piy]
				c2 = b.Li_tot[j,pix,piy]
				A[i,j][pix,piy] = H[c1,c2]
				# if i==4 && j==4
					# px(H[c1,c2])
				# end
			end
		end
	end
	pl = plotVeff(A,f,b)
	savefig(pl,"H.png")
end

######################## Misc

# if the directory path doesn't exist, it creates it, otherwise does nothing
create_dir(path) = if !isdir(path) mkdir(path) end

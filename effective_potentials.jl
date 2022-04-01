using Plots, LinearAlgebra, JLD, FFTW
px = println
include("common_functions.jl")
include("misc/create_bm_pot.jl")

################## Imports graphene.jl quantities (u1, u2, V, Vint) and produces effective potentials without plotting and tests

function import_and_computes(N,Nz)
	p = EffPotentials()
	import_u1_u2_V(N,Nz,p)
	import_Vint(p)
	init_EffPot(p)
	build_blocks_potentials(p)
	build_block_𝔸(p)
	p
end

################## EffPotentials, parameters of graphene honeycomb structures

mutable struct EffPotentials
	# Cell
	a; a1; a2; a1_star; a2_star
	x_axis_cart
	dx; dS; dz; dv
	k_axis; k_grid
	kz_axis
	K_red
	N2d; N3d
	N; Nz; L; interlayer_distance
	cell_area; Vol

	# Quantities computed and exported by graphene.jl
	v_f; u1_f; u2_f; u1v_f; u2v_f; prods_f; Vint_f
	v_dir; u1_dir; u2_dir; u1v_dir; u2v_dir; prods_dir; Vint_dir

	# Effective potentials
	Σ # Σ = <<u_j, u_{j'}>>^+-
	𝕍_V; 𝕍_Vint; 𝕍 # 𝕍_V = <<u_j, V u_{j'}>>^+-, 𝕍_Vint = <<u_j, Vint u_{j'}>>^+-, 𝕍 = 𝕍_V + 𝕍_Vint
	Wplus; W_Vint_matrix; W # Wplus = <<V, u_j u_{j'}>>^++, W_Vint_matrix = <u_j,Vint u_{j'}>, W = Wplus + W_Vint_matrix
	𝔸1; 𝔸2; 𝔹1; 𝔹2

	# Plots
	plots_cutoff
	plots_res # resolution, plots_res × plots_res
	plots_n_motifs # roughly the number of periods we see on plots
	root_path
	
	function EffPotentials()
		p = new()
		p
	end
end

function init_EffPot(p)
	init_cell_vectors(p)
	init_cell_infinitesimals(p)
	p.root_path = "effective_potentials/"
	create_dir(p.root_path)
end

################## Core: computation of effective potentials

# Builds the Fourier coefficients 
# C_m = ∑_M conj(hat(g))_{m,M} hat(f)_{m,M} e^{i d q_M 2π/L}
function build_Cm(g,f,p) 
	C = zeros(ComplexF64,p.N,p.N)
	app(kz) = cis(2*p.interlayer_distance*kz*π/p.L)
	expo = app.(p.kz_axis)
	p.L*[sum(conj.(g[m,n,:]).*f[m,n,:].*expo) for m=1:p.N, n=1:p.N]
end

function div_three(m,n,p) # Test whether [m;n] is in 3ℤ^2, returns [m;n]/3 if yes
	Q = 3
	A = mod(m,Q); B = mod(n,Q)
	bool = (A == 0 && B == 0)
	if bool
		return (bool,Int(m/Q),Int(n/Q)) # in coordinates [0,N] !! so don't do mod1 here
	else
		return (bool,0,0)
	end
end

# 2 × 2 matrix, magnetic ∈ {"no","1","2"}, f and g are in Fourier. transl is for diagonal blocks
function build_potential(g,f,p;magnetic="no",transl=true) 
	C = build_Cm(g,f,p)
	P = zeros(ComplexF64,p.N,p.N)
	# Fills P_n^D = C^D_{F^{-1} J^{*,-1} F(n)}
	(m3K,n3K) = Tuple(Int.(3*p.K_red))
	for m=1:p.N
		for n=1:p.N
			# px("TYPE ",p.k_grid)
			(m0,n0) = p.k_grid[m,n]
			(m1,n1) = transl ? (m0-m3K,n0-n3K) : (m0,n0)
			(B,m2,n2) = div_three(m1,n1,p)
			if !B
				P[m,n] = 0
			else
				K = m0*p.a1_star .+ n0*p.a2_star
				magn_fact = magnetic == "no" ? 1 : (1/3)*(magnetic == "1" ? K[1] : K[2])
				(m3,n3) = k_inv(m2,n2,p)
				P[m,n] = magn_fact*C[m3,n3]
			end
		end
	end
	return P
end

function compute_W_Vint_term(p) # matrix <u_j, Vint u_{j'}>
	M = zeros(ComplexF64,2,2)
	for mx=1:p.N
		for my=1:p.N
			for nz1=1:p.N
				for nz2=1:p.N
		dmz = kz_inv(nz1-nz2,p)
		M[1,1] += conj(p.u1_f[mx,my,nz1])*p.u1_f[mx,my,nz2]*p.Vint_f[dmz]
		M[1,2] += conj(p.u1_f[mx,my,nz1])*p.u2_f[mx,my,nz2]*p.Vint_f[dmz]
		M[2,1] += conj(p.u2_f[mx,my,nz1])*p.u1_f[mx,my,nz2]*p.Vint_f[dmz]
		M[2,2] += conj(p.u2_f[mx,my,nz1])*p.u2_f[mx,my,nz2]*p.Vint_f[dmz]
				end
			end
		end
	end
	p.W_Vint_matrix = M*p.dv/p.N^2
end

################## Computation of blocks

function build_block_𝔸(p)
	p.𝔸1 = [build_potential(p.u1_f,p.u1_f,p;magnetic="1"), build_potential(p.u1_f,p.u2_f,p;magnetic="1"),
		build_potential(p.u2_f,p.u1_f,p;magnetic="1"), build_potential(p.u2_f,p.u2_f,p;magnetic="1")]
	p.𝔸2 = [build_potential(p.u1_f,p.u1_f,p;magnetic="2"), build_potential(p.u1_f,p.u2_f,p;magnetic="2"),
		build_potential(p.u2_f,p.u1_f,p;magnetic="2"), build_potential(p.u2_f,p.u2_f,p;magnetic="2")]
	K = p.K_red[1]*p.a1_star.+p.K_red[2]*p.a2_star
	p.𝔹1 = p.𝔸1 .- K[1]*p.Σ
	p.𝔹2 = p.𝔸2 .- K[2]*p.Σ
end

function build_blocks_potentials(p)
	# Computes blocks without Vint
	p.𝕍_V = [build_potential(p.u1v_f,p.u1_f,p), build_potential(p.u1v_f,p.u2_f,p),
		 build_potential(p.u2v_f,p.u1_f,p), build_potential(p.u2v_f,p.u2_f,p)]
	p.Wplus = [build_potential(p.v_f,p.prods_f[1],p;transl=false), build_potential(p.v_f,p.prods_f[2],p;transl=false),
		   build_potential(p.v_f,p.prods_f[3],p;transl=false), build_potential(p.v_f,p.prods_f[4],p;transl=false)]
	# p.W_moins = [build_potential(p.prods_f[1],p.v_f,p;η=-1,transl=false), build_potential(p.prods_f[2],p.v_f,p;η=-1,transl=false),
	# build_potential(p.prods_f[3],p.v_f,p;η=-1,transl=false), build_potential(p.prods_f[4],p.v_f,p;η=-1,transl=false)]
	p.Σ = [build_potential(p.u1_f,p.u1_f,p), build_potential(p.u1_f,p.u2_f,p),
	       build_potential(p.u2_f,p.u1_f,p), build_potential(p.u2_f,p.u2_f,p)]

	# Computes blocks with Vint
	u1Vint_f = fft(p.u1_dir.*p.Vint_dir)
	u2Vint_f = fft(p.u2_dir.*p.Vint_dir)
	compute_W_Vint_term(p)
	p.𝕍_Vint = [build_potential(p.u1_f,u1Vint_f,p), build_potential(p.u1v_f,u2Vint_f,p),
		    build_potential(p.u2v_f,u1Vint_f,p), build_potential(p.u2v_f,u2Vint_f,p)]

	# Sums the blocks with and without Vint
	p.𝕍 = p.𝕍_V .+ p.𝕍_Vint
	p.W = add_cst_block(p.Wplus,p.W_Vint_matrix,p)
end

function add_cst_block(B,cB,p) # B in Fourier so needs to add to the first component
	nB = copy(B)
	for j=1:4
		nB[j][1,1] += cB[j]
	end
	nB
end

################## Operations on functions

function rot_A(B1,B2,p) # applies R_{-2π/3} to the vector [B1,B2], where Bj contains matrices N×N
	R = rotM(-2π/3)
	A1 = similar(B1); A2 = similar(B2)
	for K=1:p.N
		for P=1:p.N
			c = R*[B1[K,P],B2[K,P]]
			A1[K,P] = c[1]; A2[K,P] = c[2]
		end
	end
	(A1,A2)
end

################## Operations on block functions

norm_block(B) = sum(abs.(B[1])) + sum(abs.(B[2])) + sum(abs.(B[3])) + sum(abs.(B[4]))
app_block(map,B,p) = [map(B[1],p),map(B[2],p),map(B[3],p),map(B[4],p)]
σ1_B_σ1(B) = [B[4],B[3],B[2],B[1]]
hermitian_block(B) = conj.([B[1],B[3],B[2],B[4]])
# hermitian_block(B) = [conj.(B[1]),conj.(B[3]),conj.(B[2]),conj.(B[4])]
U_B_U_star(B) = [B[1],cis(2π/3).*B[2],cis(4π/3).*B[3],B[4]]

# Rotations on magnetic blocks, as a vector
function rot_block(B1,B2,p)
	A1 = similar(B1); A2 = similar(B2)
	for j=1:4
		(A1[j],A2[j]) = rot_A(B1[j],B2[j],p)
	end
	(A1,A2)
end

function weight(M,α,β) # M in matrix form, applies weights α and β
	N = size(M[1,1],1)
	m = zeros(ComplexF64,N,N)
	S = [copy(m) for i=1:2, j=1:2]
	S[1,1] = α*M[1,1]; S[1,2] = β*M[1,2]; S[2,1] = β*M[2,1]; S[2,2] = α*M[2,2]
	S
end

# from potentials in matrix form to potentials in vector form
mat2lin(M) = [M[1,1],M[1,2],M[2,1],M[2,2]]

# from potentials in vector form to potentials in matrix form
function lin2mat(M)
	N = size(M[1],1)
	m = zeros(ComplexF64,N,N)
	T = [copy(m) for i=1:2, j=1:2]
	T[1,1] = M[1]; T[1,2] = M[2]; T[2,1] = M[3]; T[2,2] = M[4]
	T
end

################## Symmetry tests

function test_particle_hole_block(B,p;name="B")
	PB_four = hermitian_block(B) # parity ∘ conj in direct becomes just conj in Fourier
	px("Test ",name,"(-x)^* = ",name,"(x) ",relative_distance_blocks(B,PB_four))
end

function test_PT_block(B,p;name="B")
	B_direct = ifft.(B)
	σ1Bσ1 = σ1_B_σ1(B_direct)
	symB = conj.(app_block(parity_x,σ1Bσ1,p))
	px("Test σ1 conj(",name,")(-x) σ1 = ",name,"(x) ",relative_distance_blocks(B_direct,symB))
end

function test_R_block(B,p;name="B")
	RB = app_block(R_four,B,p)
	px("Test R ",name," = U",name,"U* ",relative_distance_blocks(U_B_U_star(B),RB))
end

function test_R_magnetic_block(B1,B2,p;name="B")
	RB1 = app_block(R_four,B1,p)
	RB2 = app_block(R_four,B2,p)
	U_B1_Ustar = U_B_U_star(B1)
	U_B2_Ustar = U_B_U_star(B2)
	(R_U_B1_Ustar,R_U_B2_Ustar) = rot_block(U_B1_Ustar,U_B2_Ustar,p)
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

function test_build_potential_direct(g,f,p) # by P(x)= = ∑_m Cm e^{i2πxJ^*m}, used to test because it's much heavier than building by Fourier. PERIOD L, NOT L/2 !!
	C = build_Cm(g,f,p)
	P = zeros(ComplexF64,p.N,p.N)
	calJ_star = [1 -2;2 -1]
	calJ_m = [calJ_star*[p.k_grid[m,n][1];p.k_grid[m,n][2]] for m=1:p.N, n=1:p.N]
	for x=1:p.N
		for y=1:p.N
			expos = [exp(im*2π*(p.x_axis_cart[x]*calJ_m[m1,m2][1]+p.x_axis_cart[y]*calJ_m[m1,m2][2])) for m1=1:p.N, m2=1:p.N]
			P[x,y] = sum(C.*expos)
		end
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
	D = ifft.(B)
	px("In direct")
	test_equality_blocks_interm(D,p)
	px("In Fourier")
	test_equality_blocks_interm(B,p)
end

function relative_distance_blocks(B,C)
	count = 0; tot = 0
	for i=1:4
		count += sum(abs.(B[i].-C[i]))
		tot += sum(abs.(B[i]))
	end
	count/tot
end

function test_block_hermitianity(C,p;name="")
	B = ifft.(C)
	c = 0; T = 0
	for x=1:p.N
		for y=1:p.N
			h = [B[1][x,y] B[2][x,y]; B[3][x,y] B[4][x,y]]
			c += sum(abs.(h.-h'))
			T += sum(abs.(h))
		end
	end
	px("Test block hermitianity ",name," ",c/T)
end

################## Import functions

function import_u1_u2_V(N,Nz,p)
	p.N = N; p.Nz = Nz
	path = "graphene/exported_functions/"
	f = string(path,"N",N,"_Nz",Nz,"_u1_u2_V.jld")

	p.a = load(f,"a"); p.L = load(f,"L")
	init_cell_infinitesimals(p)

	p.v_f = load(f,"v_f")
	p.u1_f = load(f,"u1_f")
	p.u2_f = load(f,"u2_f")
	p.u1v_f = load(f,"vu1_f")
	p.u2v_f = load(f,"vu2_f")
	p.prods_f = load(f,"prods_f")

	p.v_dir = ifft(p.v_f)
	p.u1_dir = ifft(p.u1_f)
	p.u2_dir = ifft(p.u2_f)
	p.u1v_dir = ifft(p.u1v_f)
	p.u2v_dir = ifft(p.u2v_f)
	p.prods_dir = ifft.(p.prods_f)
end

function import_Vint(p)
	path = "graphene/exported_functions/"
	f = string(path,"N",p.N,"_Nz",p.Nz,"_Vint.jld")
	p.interlayer_distance = load(f,"interlayer_distance")
	a = load(f,"a"); L = load(f,"L"); @assert a==p.a && L==p.L

	Vint_f = load(f,"Vint_f")
	p.Vint_f = zeros(ComplexF64,p.N,p.N,p.Nz)
	p.Vint_f[1,1,:] = Vint_f
	p.Vint_dir = ifft(p.Vint_f)
end

################## Plot functions

# creates the function of reduced direct space from the array in reduced Fourier space
function red_arr2red_fun(ϕ_four_red,p) 
	a(x,y) = 0
	for i=1:p.N
		ki = p.k_axis[i]
		if abs(ki) <= p.plots_cutoff
			for j=1:p.N
				kj = p.k_axis[j]
				if abs(kj) <= p.plots_cutoff
					c(x,y) = ϕ_four_red[i,j] * cis(2π*(ki*x+kj*y))
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


function eval_fun_to_plot(f_four,fun,n_motifs,res,p)
	# Computes function in cartesian space
	fu = red_arr2red_fun(f_four,p)
	ψ2 = red2cart_function(fu,p)
	f = scale_fun2d(ψ2,n_motifs)

	# Evaluates
	fun.([f(i/res,j/res) for i=0:res-1, j=0:res-1])
end

# B is a 2 × 2 matrix of potentials
# from array of Fourier coefficients to plot in direct cartesian space
function plot_block_cart(B_four,p;title="plot_full") 
	path = string(p.root_path,"plots_potentials_cartesian/")
	create_dir(path)
	funs = [real,imag,abs]; titles = ["real","imag","abs"]
	for I=1:3
		h = []
		for m=1:4
			ψ_ar = eval_fun_to_plot(B_four[m],funs[I],p.plots_n_motifs,p.plots_res,p)
			hm = heatmap(ψ_ar,size=(300,200))
			# hm = heatmap(ψ_ar,aspect_ratio=:equal)
			push!(h,hm)
		end
		size = 1000
		pl = plot(h...,layout=(2,2),size=(1300,1000),legend=false)
		savefig(pl,string(path,title,"_",titles[I],"_cart.png"))
		px("Plot of ",title," ",titles[I]," in cartesian coords, done")
	end
end

# other_magnetic_block is the additional magnetic block, in this case it plots |B1|^2+|B2|^2
function plot_magnetic_block_cart(B1_four,B2_four,p;title="plot_full") 
	h = []
	path = string(p.root_path,"plots_potentials_cartesian/")
	create_dir(path)
	for m=1:4
		ψ_ar = eval_fun_to_plot(B1_four[m],abs2,p.plots_n_motifs,p.plots_res,p)
		ψ_ar .+= eval_fun_to_plot(B2_four[m],abs2,p.plots_n_motifs,p.plots_res,p)
		hm = heatmap(ψ_ar,size=(300,200))
		# hm = heatmap(ψ_ar,aspect_ratio=:equal)
		push!(h,hm)
	end
	size = 1000
	pl = plot(h...,layout=(2,2),size=(1300,1000),legend=false)
	savefig(pl,string(path,"abs2_",title,"_cart.png"))
	px("Plot of |",title,"|^2 in cartesian coords, done")
end

function plot_block_reduced(B,p;title="plot_full")
	four = [true,false]
	funs = [real,imag,abs]; titles = ["real","imag","abs"]
	path = string(p.root_path,"plots_potentials_reduced/")
	create_dir(path)
	for fo=1:2
		for I=1:3
			h = []
			for m=1:4
				a = funs[I].(four[fo] ? B[m] : ifft(B[m]))
				hm = heatmap(a,size=(200,200),aspect_ratio=:equal)
				push!(h,hm)
			end
			size = 1000
			pl = plot(h...,layout=(2,2),size=(1300,1000),legend=false)
			savefig(pl,string(path,title,"_",titles[I],"_",four[fo] ? "four" : "dir",".png"))
			# px("Plot of ",title," ",titles[I]," done")
		end
	end
end

# Regarder d grand

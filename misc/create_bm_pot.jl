# using LinearAlgebra
fill2d(x,n) = [copy(x) for i=1:n, j=1:n]

function Ts(α,β) # α,β = 1,0 is the anti-chiral model, α,β = 0,1 is the chiral one (with vanishing bands)
	σ0 = [1 0;0 1]; σ1 = [0 1; 1 0]; σ2 = [0 -im; im 0]
	T1 = α*σ0 + β*σ1
	T2 = α*σ0 + β*(-(1/2)*σ1 + (sqrt(3)/2)*σ2)
	T3 = α*σ0 + β*(-(1/2)*σ1 - (sqrt(3)/2)*σ2)
	# ϕ = 2π/3
	# T1 = β*[0 1;1 0] + α*I
	# T2 = β*[0 cis(-ϕ); cis(ϕ) 0] + α*I
	# T3 = β*[0 cis(ϕ); cis(-ϕ) 0] + α*I
	(T1,T2,T3)
end


function build_BM_objects(a,α,β)
	# Conventions of Becker, Embree, Wittsten, Zworski
	red2cart = (a/2)*[sqrt(3) sqrt(3);1 -1]; cart2red = inv(red2cart) # matrix to pass from reduce coordinates (in the basis (a1,a2)) to cartesian coordinates, X_cart = red2cart*X_red, and its inverse
	# q[1] = [0,-1]; q[2] = (1/2)*[sqrt(3),1]; q[3] = (1/2)*[-sqrt(3),1]
	(T1,T2,T3) = Ts(α,β)
	q = [copy([0.0,0.0]) for i=1:3]

	# One orientation
	# a1_star = (2pi/a)*[1/sqrt(3), 1]
	# a2_star = (2pi/a)*[1/sqrt(3),-1]
	# q[3] = a1_star; q[2] = a2_star; q[1] = (4π/(a*sqrt(3)))*[-1,0]
	# Another orientation
	q1 = [0 -1]; q2 = [sqrt(3)/2 1/2]; q3 = [-sqrt(3)/2 1/2]
	q = (4π/(a*sqrt(3)))*[q1,q2,q3]
	J = [0 -1;1 0]
	# q = [J*q[i] for i=1:3]

	Tcart(r) = T1*cis(-q[1]⋅r) + T2*cis(-q[2]⋅r) + T3*cis(-q[3]⋅r) # in cartesian coordinates
	Tred(r) = Tcart(red2cart*r) # in reduced coordinates
	(Tred,Tcart)
end

####################### Direct or Fourier

function T_BM(N,a,α,β)
	m = zeros(ComplexF64,N,N)
	T = fill2d(m,2)
	(Tfun,Tfun_cart) = build_BM_objects(a,α,β)
	for sx=1:N
		for sy=1:N
			Ts = Tfun([sx,sy]./N)
			for i=1:2
				for j=1:2
					T[i,j][sx,sy] = Ts[i,j]
				end
			end
		end
	end
	T
end


function T_BM_four(N,α,β;scale=false)
	kv(k) = Int(mod(k,N))+1
	(T1,T2,T3) = Ts(α,β)
	m = zeros(ComplexF64,N,N)
	T = fill2d(m,2)

	q1 = [-1;-1]; q2 = [0;1]; q3 = [1;0]
	L = scale ? -[1 -2;2 -1] : [1 0;0 1]
	# Lq : L q1 = [-1,1], L q2 = [2,1], L q3 = [-1,-2]
	q1,q2,q3 = L*q1,L*q2,L*q3
	q1i1,q1i2 = kv(q1[1]),kv(q1[2])
	q2i1,q2i2 = kv(q2[1]),kv(q2[2])
	q3i1,q3i2 = kv(q3[1]),kv(q3[2])
	for i=1:2
		for j=1:2
			# T[i,j][2,2] = T1[i,j]; T[i,j][1,end] = T2[i,j]; T[i,j][end,1] = T3[i,j]
			T[i,j][q1i1,q1i2] = T1[i,j]
			T[i,j][q2i1,q2i2] = T2[i,j]
			T[i,j][q3i1,q3i2] = T3[i,j]
		end
	end
	T
end

function build_BM(α,β,p;scale=false)
	T = T_BM_four(p.N,α,β;scale=scale)
	[T[1,1],T[1,2],T[2,1],T[2,2]]
end

####################### Builds the 4×4 matrix

function Veff_BM(N,a,α=1,β=1,fourier=true)
	m = zeros(ComplexF64,N,N); V = fill2d(m,4)
	T = fourier ? T_BM_four(N,a,α,β) : T_BM(N,a,α,β)
	for i=1:2
		for j=1:2
			V[i,j+2] = T[i,j]
		end
	end
	for i=3:4
		for j=1:2
			V[i,j] = conj.(V[j,i]) # and not V[j,i]'
		end
	end
	V
end

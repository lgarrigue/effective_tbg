# using LinearAlgebra
fill2d(x,n) = [copy(x) for i=1:n, j=1:n]

function Ts(α,β) # α,β = 1,0 is the anti-chiral model, α,β = 0,1 is the chiral one (with vanishing bands)
	# Form taken from Becker, Embree, Wittsen, Zworski : Spectral characterization of magic angles in TBG, eq (15) and after
	σ0 = [1 0;0 1]; σ1 = [0 1; 1 0]; σ2 = [0 -im; im 0]
	T1 = α*σ0 + β*σ1
	T2 = α*σ0 + β*(-(1/2)*σ1 + (sqrt(3)/2)*σ2)
	T3 = α*σ0 + β*(-(1/2)*σ1 - (sqrt(3)/2)*σ2)
	τ = cis(2π/3)
	@assert T1 == β*[0 1;1 0] + α*I
	@assert sum(abs.(T2 .- β*[0 τ^2; τ 0] .- α*σ0))<1e-10
	@assert sum(abs.(T3 .- β*[0 τ; τ^2 0] .- α*σ0))<1e-10
	(T1,T2,T3)
end

####################### Direct or Fourier

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
			# T[i,j][end,end] = T1[i,j]; T[i,j][2,1] = T2[i,j]; T[i,j][1,2] = T3[i,j]
			# T[i,j][2,2] = T1[i,j]; T[i,j][3,end] = T2[i,j]; T[i,j][end,3] = T3[i,j]
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

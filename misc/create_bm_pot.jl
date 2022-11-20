fill2d(x,n) = [copy(x) for i=1:n, j=1:n]
block2ar(T) = [T[1,1],T[1,2],T[2,1],T[2,2]]

function T_matrices(α,β) # α,β = 1,0 is the anti-chiral model, α,β = 0,1 is the chiral one (with vanishing bands)
	# Form taken from Becker, Embree, Wittsen, Zworski : Spectral characterization of magic angles in TBG, eq (15) and after
	σ0 = [1 0;0 1]; σ1 = [0 1; 1 0]; σ2 = [0 -im; im 0]
	T1 = α*σ0 + β*σ1
	T2 = α*σ0 + β*(-(1/2)*σ1 + (sqrt(3)/2)*σ2)
	T3 = α*σ0 + β*(-(1/2)*σ1 - (sqrt(3)/2)*σ2)
	τ = cis(2π/3)
	@assert T1 == β*[0 1;1 0] + α*I
	@assert sum(abs.(T2 .- β*[0 τ^2; τ 0] .- α*σ0))<1e-10
	@assert sum(abs.(T3 .- β*[0 τ; τ^2 0] .- α*σ0))<1e-10
	(block2ar(T1),block2ar(T2),block2ar(T3))
end

function T_BM_four(α,β,p;second=false)
        δ = -1*p.gauge_param
        (T1,T2,T3) = T_matrices(α,δ*β)
	m = zeros(ComplexF64,p.N,p.N)
	T = [copy(m) for i=1:4]
	q1 = [0;0]; q2 = [-1;0]; q3 = [0;-1]
	if second # following Fourier coeffcients to build BM Hamiltonians
		q1 = [-1;-1]; q2 = [1;-1]; q3 = [-1;1]
	end
	q1i1,q1i2 = k_inv_v(q1,p)
        q2i1,q2i2 = k_inv_v(q2,p)
        q3i1,q3i2 = k_inv_v(q3,p)
	for i=1:4
		T[i][q1i1,q1i2] = T1[i]
		T[i][q2i1,q2i2] = T2[i]
		T[i][q3i1,q3i2] = T3[i]
	end
	T
end

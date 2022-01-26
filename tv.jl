# Jonas August, 1/2021

"Denoises image v by minimizing DTV(u) + λ|u-v|^2 using the 'digital total variation' proposed by Chan, Osher, and Shen (COS) in 'The Digital TV Filter and Nonlinear Denoising', Trans. IEEE Image Process, Feb. 2001.  Returns sequence of iterates."
function chan_osher_shen_tv(v::Array{T,2};λ::S=S(8),itermax::Int=100,devmin²::S=S((2^-10)^2)) where {S<:AbstractFloat,T<:Union{S,Gray{S}}}  

    # Pixels should be a floating point type to avoid artifacts, e.g., subtraction a-b where a<b, a>0, b>0.
    # devmin² could be even smaller (e.g., eps(T)^2), but that would require subnormal numbers which may be slow.
    # Also, they might be flushed to zero, depending on details.  I'm mostly only assuming 0 < devmin² << 1, and
    # especially 0 < sqrt(devmin²).

    m,n = size(v)
    u = zeros(eltype(v),0:m+1,0:n+1)  # pad by 1
    u[1:m,1:n] .= v  # Initial condition
    
    # Enforce 0 outward derivative for u BC.
    u[0,:] .= u[1,:];  u[:,0] .= u[:,1];  u[end,:] .= u[end-1,:];  u[:,end] .= u[:,end-1]
    
    # devinv = 1 / local deviation
    devinv = similar(u)  
    
    # Dummy BCs for devinv (ultimately ignored).
    devinv[0,:] .= 1;  devinv[:,0] .= 1;  devinv[end,:] .= 1;  devinv[:,end] .= 1

    us = zeros(eltype(v),m,n,itermax)  # TODO: grow history instead of preallocate?
    for iter in 1:itermax
        # 1/(local deviation) image is precomputed for speedup, so not
        # exactly the same behavior as in the COS paper.
        # TODO: don't precompute, so that we have exact COS behavior.  Timing?
        @inbounds for i=1:m, j=1:n
            devinv[i,j] = 1/sqrt(devmin² + sum(y->y^2,(u[i+1,j],u[i,j+1],u[i-1,j],u[i,j-1]) .- u[i,j]))
        end
        
        @inbounds for i=1:m, j=1:n 
            we = devinv[i,j+1]+devinv[i,j]
            wn = devinv[i-1,j]+devinv[i,j]
            ww = devinv[i,j-1]+devinv[i,j]
            ws = devinv[i+1,j]+devinv[i,j]
            
            # Kill weights along boundary  
            # TODO: not much speedup if we pull this BC handling outside i,j loop? looks less than 20% time vs. no BC
            wn = (i == 1) ? 0 : wn
            ws = (i == m) ? 0 : ws
            ww = (j == 1) ? 0 : ww
            we = (j == n) ? 0 : we
   
            u[i,j] = (λ*v[i,j] + we*u[i,j+1] + wn*u[i-1,j] + ww*u[i,j-1] + ws*u[i+1,j])/(λ+we+wn+ww+ws)
        end

        # Enforce 0 outward derivative for u BC.
        u[0,:] .= u[1,:]; u[:,0] .= u[:,1]; u[end,:] .= u[end-1,:]; u[:,end] .= u[:,end-1]
    
        us[:,:,iter] = u[1:m,1:n]
    end
    us  # sequence of iterations is returned
end

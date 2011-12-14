function DpEz = get_DpEz(N, Nx, Ny, Pp, BC, mat_dp_prim)

const;

DpEz = sparse(N,N);

% Every Ez component participates in two p-derivatives.
% In one curl loop it is substracted (-Ez), and in the other loop it is added (+Ez).

% -Ez
negEz = -ones(Nx,Ny);

% On p = 0 plane
% Do not handle BC(Pp,Neg) == Bloch here.  That case is handled by BC(Pp,Pos) == Bloch.
% Note that BC(Pp,Neg) == Bloch implies BC(Pp,Pos) == Bloch.
if BC(Pp,Neg) == PEC
    if Pp == Xx
        negEz(1,:) = 0;
    else 
        assert(Pp == Yy);
        negEz(:,1) = 0;
    end
end

DpEz = spdiags(negEz(:), 0, DpEz);  % 0 means -Ez(x,y) is used to calculate Hz(x+-0,y).

% +Ez
posEz = ones(Nx,Ny);

% On p = Np plane
if BC(Pp,Pos) == PEC || BC(Pp,Pos) == Bloch  % Note that actually there are no other cases, since BC(Pp,Pos) cannot be PMC.
    if Pp == Xx
        posEz(1,:) = 0;
    else
        assert(Pp == Yy)
        posEz(:,1) = 0;
    end
end

if Pp == Xx
    DpEz = spdiags(posEz(:), 1, DpEz);  % 1 means +Ez(x,y) is used to calculate Hz(x-1,y).
else
    assert(Pp == Yy)
    DpEz = spdiags(posEz(:), Nx, DpEz);  % Nx means +Ez(x,y) is used to calculate Hz(x,y-1).
end
    

if BC(Pp,Pos) == Bloch
    if Pp == Xx
        for j = 1:Ny
            DpEz(j*Nx,(j-1)*Nx+1) = DpEz(j*Nx,(j-1)*Nx+1) + 1;
        end
    else
        for i = 1:Nx
            DpEz((Ny-1)*Nx+i,i) = DpEz((Ny-1)*Nx+i,i) + 1;
        end
    end
end

DpEz = mat_dp_prim \ DpEz;  % To calculate the curl, Ep shuold be divided by the primary edge lengths, which are centered at dual grid vertices.
function DpEp = get_DpEp(N, Nx, Ny, Pp, BC, mat_dp_dual)

const;

DpEp = sparse(N,N);

% Every Ep component participates in two p-derivatives.
% In one derivative it is substracted (-Ep), and in the other derivative it is added (+Ep).

% +Ep
posEp = ones(Nx,Ny);

% PEC boundary condition on p = 0 plane
% Do not handle BC(Pp,Neg) == Bloch here.  That case is handled by BC(Pp,Pos) == Bloch.
% Note that BC(Pp,Neg) == Bloch implies BC(Pp,Pos) == Bloch.
if BC(Pp,Neg) == PEC
    if Pp == Xx
        posEp(1,:) = 0;
    else 
        assert(Pp == Yy);
        posEp(:,1) = 0;
    end
end

if BC(Pp,Neg) == PMC
    if Pp == Xx
        posEp(1,:) = 2;
    else 
        assert(Pp == Yy);
        posEp(:,1) = 2;
    end
end

DpEp = spdiags(posEp(:), 0, DpEp);  % 0 means +Ep(x,y) is used to calculate rho(x,y).

% -Ep
negEp = -ones(Nx,Ny);

% PEC and periodic boundary condition on p = Np plane
if BC(Pp,Pos) == PEC || BC(Pp,Pos) == Bloch  % Note that actually there are no other cases, since BC(Pp,Pos) cannot be PMC.
    if Pp == Xx
        negEp(end,:) = 0;
    else
        assert(Pp == Yy)
        negEp(:,end) = 0;
    end
end

if Pp == Xx
    DpEp = spdiags(negEp(:), -1, DpEp);  % -1 means -Ep(x,y) is used to calculate rho(x+1,y).
else
    assert(Pp == Yy)
    DpEp = spdiags(negEp(:), -Nx, DpEp);  % -Nx means -Ep(x,y) is used to calculate rho(x,y+1).
end
    

if BC(Pp,Pos) == Bloch
    if Pp == Xx
        for j = 1:Ny
            DpEp((j-1)*Nx+1,j*Nx) = DpEp((j-1)*Nx+1,j*Nx) - 1;
        end
    else
        for i = 1:Nx
            DpEp(i,(Ny-1)*Nx+i) = DpEp(i,(Ny-1)*Nx+i) - 1;
        end
    end
end

DpEp = mat_dp_dual \ DpEp;  % To calculate the curl, Ep shuold be divided by the primary edge lengths, which are centered at dual grid vertices.
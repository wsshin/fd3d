function DpEq = get_DpEq(N, Nx, Ny, Pp, BC, mat_dp_prim)

const;

DpEq = sparse(N,N);

% Every Eq component participates in two p-derivatives.
% In one curl loop it is substracted (-Eq), and in the other loop it is added (+Eq).

% -Eq
negEq = -ones(Nx,Ny);

% On p = 0 plane
% Do not handle BC(Pp,Neg) == Bloch here.  That case is handled by BC(Pp,Pos) == Bloch.
% Note that BC(Pp,Neg) == Bloch implies BC(Pp,Pos) == Bloch.
if BC(Pp,Neg) == PEC
    if Pp == Xx
        negEq(1,:) = 0;
    else 
        assert(Pp == Yy);
        negEq(:,1) = 0;
    end
end

% On q = 0 plane
if BC(Qq,Neg) == PMC
    if Qq == Xx
        negEq(:,1) = 0;
    else 
        assert(Qq == Yy);
        negEq(1,:) = 0;
    end
end


DpEq = spdiags(negEq(:), 0, DpEq);  % 0 means -Eq(x,y) is used to calculate Hz(x+-0,y).

% +Eq
posEq = ones(Nx,Ny);

% On p = Np plane
if BC(Pp,Pos) == PEC || BC(Pp,Pos) == Bloch  % Note that actually there are no other cases, since BC(Pp,Pos) cannot be PMC.
    if Pp == Xx
        posEq(1,:) = 0;
    else
        assert(Pp == Yy)
        posEq(:,1) = 0;
    end
end

if Pp == Xx
    DpEq = spdiags(posEq(:), 1, DpEq);  % 1 means +Eq(x,y) is used to calculate Hz(x-1,y).
else
    assert(Pp == Yy)
    DpEq = spdiags(posEq(:), Nx, DpEq);  % Nx means +Eq(x,y) is used to calculate Hz(x,y-1).
end
    

if BC(Pp,Pos) == Bloch
    if Pp == Xx
        for j = 1:Ny
            DpEq(j*Nx,(j-1)*Nx+1) = DpEq(j*Nx,(j-1)*Nx+1) + 1;
        end
    else
        for i = 1:Nx
            DpEq((Ny-1)*Nx+i,i) = DpEq((Ny-1)*Nx+i,i) + 1;
        end
    end
end

DpEq = mat_dp_prim \ DpEq;  % To calculate the curl, Ep shuold be divided by the primary edge lengths, which are centered at dual grid vertices.
function DpHp = get_DpHp(N, Nx, Ny, Pp, BC, mat_dp_prim)

const;

DpHp = sparse(N,N);

% Every Hp component participates in two p-derivatives.
% In one derivative it is substracted (-Hp), and in the other derivative it is added (+Hp).

% -Hp
negHp = -ones(Nx,Ny);

% PEC boundary condition on p = 0 plane
% Do not handle BC(Pp,Neg) == Bloch here.  That case is handled by BC(Pp,Pos) == Bloch.
% Note that BC(Pp,Neg) == Bloch implies BC(Pp,Pos) == Bloch.
if BC(Pp,Neg) == PEC
    if Pp == Xx
        negHp(1,:) = 0;
    else 
        assert(Pp == Yy);
        negHp(:,1) = 0;
    end
end

DpHp = spdiags(negHp(:), 0, DpHp);  % 0 means -Hp(x,y) is used to calculate Hz(x+-0,y).

% +Hp
posHp = ones(Nx,Ny);

% PEC and periodic boundary condition on p = Np plane
if BC(Pp,Pos) == PEC || BC(Pp,Pos) == Bloch  % Note that actually there are no other cases, since BC(Pp,Pos) cannot be PMC.
    if Pp == Xx
        posHp(1,:) = 0;
    else
        assert(Pp == Yy)
        posHp(:,1) = 0;
    end
end

if Pp == Xx
    DpHp = spdiags(posHp(:), 1, DpHp);  % 1 means +Hp(x,y) is used to calculate Hz(x-1,y).
else
    assert(Pp == Yy)
    DpHp = spdiags(posHp(:), Nx, DpHp);  % Nx means +Hp(x,y) is used to calculate Hz(x,y-1).
end
    

if BC(Pp,Pos) == Bloch
    if Pp == Xx
        for j = 1:Ny
            DpHp(j*Nx,(j-1)*Nx+1) = DpHp(j*Nx,(j-1)*Nx+1) + 1;
        end
    else
        for i = 1:Nx
            DpHp((Ny-1)*Nx+i,i) = DpHp((Ny-1)*Nx+i,i) + 1;
        end
    end
end

DpHp = mat_dp_prim \ DpHp;  % To calculate the curl, Ep shuold be divided by the primary edge lengths, which are centered at dual grid vertices.
function DpHq = get_DpHq(N, Nx, Ny, Pp, BC, mat_dp_dual)

const; 

if Pp == Xx
    Qq = Yy;
else
    assert(Pp == Yy);
    Qq = Xx;
end

DpHq = sparse(N,N);

% Every Hq component participates in two p-derivatives.
% In one curl loop it is substracted (-Hq), and in the other loop it is added (+Hq).

% +Hq
posHq = ones(Nx,Ny);  

% PEC boundary condition on q = 0 plane
% We don't need to handle the periodic or PMC boundary condition in the q-direction.
if BC(Qq,Neg) == PEC
    if Qq == Xx  % Pp == Yy
        posHq(1,:) = 0;
    else  % Pp == Xx
        assert(Qq == Yy);
        posHq(:,1) = 0;
    end
end

% PMC boundary condition on p = 0 plane
% Do not handle BC(Pp,Neg) == Bloch here.  That case is handled by BC(Pp,Pos) == Bloch.
% Note that BC(Pp,Neg) == Bloch implies BC(Pp,Pos) == Bloch.
if BC(Pp,Neg) == PMC
    if Pp == Xx
        posHq(1,:) = 2;
    else
        assert(Pp == Yy);
        posHq(:,1) = 2;
    end
end

% PEC boundary condition on p = 0 plane
% Assume that there is a symmetric Hq behind the p = 0 plane.
if BC(Pp,Neg) == PEC
    if Pp == Xx
        posHq(1,:) = 0;
    else
        assert(Pp == Yy);
        posHq(:,1) = 0;
    end
end


DpHq = spdiags(posHq(:), 0, DpHq);  % 0 means +Hq(x,y) is used to calculate Er(x,y).

% -Hq
negHq = -ones(Nx,Ny);  

% PEC boundary condition on q = 0 plane
% We don't need to handle the periodic or PMC boundary condition in the q-direction.
if BC(Qq,Neg) == PEC
    if Qq == Xx  % Pp == Yy
        negHq(1,:) = 0;
    else  % Pp == Xx
        assert(Qq == Yy);
        negHq(:,1) = 0;
    end
end

% On p = Np plane
if Pp == Xx
    negHq(Nx,:) = 0;
    DpHq = spdiags(negHq(:), -1, DpHq);  % -1 means -Hq(x,y) is used to calculate Er(x+1,y).
else
    assert(Pp == Yy);
    negHq(:,Ny) = 0;
    DpHq = spdiags(negHq(:), -Nx, DpHq);  % -Nx means -Hq(x,y) is used to calculate Ex(x,y+1).
end 

% Periodic boundary condition on p = Np plane
if BC(Pp,Pos) == Bloch
    if Pp == Xx
        for j = 0:Ny-1
            DpHq(j*Nx+1,(j+1)*Nx) = DpHq(j*Nx+1,(j+1)*Nx) - 1;
        end
    else
        assert(Pp == Yy);
        for i = 1:Nx
            DpHq(i,(Ny-1)*Nx+i) = DpHq(i,(Ny-1)*Nx+i) - 1;
        end
    end
end

DpHq = mat_dp_dual \ DpHq;  % To calculate the curl, Hq shuold be divided by the edge lengths centered at primary grid vertices.
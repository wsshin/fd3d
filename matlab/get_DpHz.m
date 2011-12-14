function DpHz = get_DpHz(N, Nx, Ny, Pp, BC, mat_dp_dual)

const; 

DpHz = sparse(N,N);

% Every Hz component participates in two x-derivatives.
% In one curl loop it is substracted (-Hz), and in the other loop it is added (+Hz).

% +Hz
posHz = ones(Nx,Ny);  

% On p = 0 plane
% Do not handle BC(Pp,Neg) == Bloch here.  That case is handled by BC(Pp,Pos) == Bloch.
% Note that BC(Pp,Neg) == Bloch implies BC(Pp,Pos) == Bloch.
if BC(Pp,Neg) == PMC
    if Pp == Xx
        posHz(1,:) = 2;
    else
        assert(Pp == Yy);
        posHz(:,1) = 2;
    end
end

DpHz = spdiags(posHz(:), 0, DpHz);  % 0 means +Hz(x,y) is used to calculate Er(x,y).

% -Hz
negHz = -ones(Nx,Ny);  

% On p = Np plane
if Pp == Xx
    negHz(Nx,:) = 0;
    DpHz = spdiags(negHz(:), -1, DpHz);  % -1 means -Hz(x,y) is used to calculate Er(x+1,y).
else
    assert(Pp == Yy);
    negHz(:,Ny) = 0;
    DpHz = spdiags(negHz(:), -Nx, DpHz);  % -Nx means -Hz(x,y) is used to calculate Ex(x,y+1).
end 

if BC(Pp,Pos) == Bloch
    if Pp == Xx
        for j = 0:Ny-1
            DpHz(j*Nx+1,(j+1)*Nx) = DpHz(j*Nx+1,(j+1)*Nx) - 1;
        end
    else
        assert(Pp == Yy);
        for i = 1:Nx
            DpHz(i,(Ny-1)*Nx+i) = DpHz(i,(Ny-1)*Nx+i) - 1;
        end
    end
end

DpHz = mat_dp_dual \ DpHz;  % To calculate the curl, Hz shuold be divided by the edge lengths centered at primary grid vertices.
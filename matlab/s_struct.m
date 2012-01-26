function s = s_struct(dim, gORd, dir, Npml_n, Npml_p, dLn, dLp, omega)

const;

s = ones(dim);
Ndir = dim(dir);

for i = 1:Ndir
    s_val = 1;
    if gORd == Prim
        if i <= Npml_n
            depth = Npml_n - (i-1);
            s_val = s_param(depth, Npml_n, dLn, omega);
        elseif i > Ndir+1-Npml_p
            depth = i - (Ndir+1-Npml_p);
            s_val = s_param(depth, Npml_p, dLp, omega);
        end
    else
        assert(gORd == Dual);
        if i <= Npml_n
            depth = Npml_n - i + 0.5;
            s_val = s_param(depth, Npml_n, dLn, omega);
        elseif i > Ndir-Npml_p
            depth = i - (Ndir-Npml_p) - 0.5;
            s_val = s_param(depth, Npml_p, dLp, omega);
        end
    end
    
    if  length(dim) == 2
        if dir == Xx
            s(i,:) = s_val;
        else
            assert(dir == Yy);
            s(:,i) = s_val;
        end
    else
        assert(length(dim) == 3);
        if dir == Xx
            s(i,:,:) = s_val;
        elseif dir == Yy
            s(:,i,:) = s_val;
        else
            assert(dir == Zz);
            s(:,:,i) = s_val;
        end
    end
end
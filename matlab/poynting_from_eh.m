function sr = poynting_from_eh(normal, intercept, e, h)
% The Poynting vectors are calculated on the points of H fields.  The 
% Poynting vectors calculated at these points multiplied with area elements 
% used for curl(E) give the correct power flux through a box bounded by 
% surfaces tangential to E fields. "intercept" is the index of the normal 
% plane tangential to E fields.

const;

[Pp Qq Rr] = cycle_axis(normal);

% Get Ep and Eq on the plane normal to "normal" at intercept.
if isa(e{Pp},'scalar3d')
    ep = e{Pp}.get_slice(normal, intercept).array;
    eq = e{Qq}.get_slice(normal, intercept).array;
else
    assert(isa(e{Pp},'scalar2d'));
    ep = e{Pp}.array;
    eq = e{Qq}.array;
end

gi = e{Pp}.gi;

Np = gi.N(Pp);
Nq = gi.N(Qq);
Nr = gi.N(Rr);

% Calculate Ep and Eq at the Hr point.
switch gi.BC(Qq,Pos)
    case PEC
        ep = [ep, zeros(Np,1)];
    case PMC
        ep = [ep, ep(:,Nq)];
    case Bloch
        ep = [ep, ep(:,1) * gi.exp_neg_ikL(Qq)];
    otherwise
        error('Not a supported boundary condition');
end

switch gi.BC(Pp,Pos)
    case PEC
        eq = [eq; zeros(1,Nq)];
    case PMC
        eq = [eq; eq(Np,:)];
    case Bloch
        eq = [eq; eq(1,:) * gi.exp_neg_ikL(Qq)];
    otherwise
        error('Not a supported boundary condition');
end

ep = (ep(:,1:end-1) + ep(:,2:end)) / 2;
eq = (eq(1:end-1,:) + eq(2:end,:)) / 2;


% Calculate Hp and Hq on the same plane as Ep and Eq, considering the staggered grid.
if isa(h{Pp},'scalar3d')
    hp = h{Pp}.get_slice(normal, intercept).array;
    hq = h{Qq}.get_slice(normal, intercept).array;
    if intercept > 1
        hp_prev = h{Pp}.get_slice(normal, intercept-1).array;
        hq_prev = h{Qq}.get_slice(normal, intercept-1).array;
    else
        switch gi.BC(Rr,Neg)
            case PEC
                hp_prev = hp;
                hq_prev = hq;
            case PMC
                hp_prev = -hp;
                hq_prev = -hq;
            case Bloch
                hp_prev = h{Pp}.get_slice(normal, Nr).array / gi.exp_neg_ikL(Rr);
                hq_prev = h{Qq}.get_slice(normal, Nr).array / gi.exp_neg_ikL(Rr);
            otherwise
                error('Not a supported boundary condition');
        end
    end
    hp = (hp + hp_prev) / 2;
    hq = (hq + hq_prev) / 2;
else
    assert(isa(h{Pp},'scalar2d'));
    hp = h{Pp}.array;
    hq = h{Qq}.array;
end

% Calculate Hp and Hq at the Hr point.
switch gi.BC(Pp,Pos)
    case PEC
        hp = [hp; zeros(1,Nq)];
    case PMC
        hp = [hp; hp(Np,:)];
    case Bloch
        hp = [hp; hp(1,:) * gi.exp_neg_ikL(Pp)];
    otherwise
        error('Not a supported boundary condition');
end

switch gi.BC(Qq,Pos)
    case PEC
        hq = [hq, zeros(Np,1)];
    case PMC
        hq = [hq, hq(:,Nq)];
    case Bloch
        hq = [hq, hq(:,1) * gi.exp_neg_ikL(Qq)];
    otherwise
        error('Not a supported boundary condition');
end

hp = (hp(1:end-1,:) + hp(2:end,:))/2;
hq = (hq(:,1:end-1) + hq(:,2:end))/2;

sr_array = real(ep.*conj(hq) - eq.*conj(hp))/2;
gk(Pp) = Dual;
gk(Qq) = Dual;
gk(Rr) = Prim;
sr = scalar2d(strcat('S',AxisName(Rr)), sr_array, normal, intercept, gk, gi, true);
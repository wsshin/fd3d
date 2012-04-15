function sr = fetch_poynting(file_base_name, normal, intercept, gi)
% The Poynting vectors are calculated on the points of H fields.  The 
% Poynting vectors calculated at these points multiplied with area elements 
% used for curl(E) give the correct power flux through a box bounded by 
% surfaces tangential to E fields. "intercept" is the index of the normal 
% plane tangential to E fields.

const;

[Pp Qq Rr] = cycle_axis(normal);

if intercept == gi.N(normal)+1
    % This works for PEC and PMC, because the time-averaged Poynting vector
    % through such boundary faces is zero.  Also works for the Bloch boundary
    % condition because the time-averaged Poynting vector does not depend on
    % phase.
    intercept = 1;
end

% Get Ep and Eq on the plane normal to "normal" at intercept.
[e h] = fetch_eh2d(file_base_name, normal, intercept, gi);

ep = e{Pp}.array;
eq = e{Qq}.array;

clear e

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
hp = h{Pp}.array;
hq = h{Qq}.array;

clear h

if intercept > 1
    [e_prev h_prev] = fetch_eh2d(file_base_name, normal, intercept-1, gi, [false false false; true true true]);
    hp_prev = h_prev{Pp}.array;
    hq_prev = h_prev{Qq}.array;
    clear e_prev h_prev
else
    switch gi.BC(Rr,Neg)
        case PEC
            hp_prev = hp;
            hq_prev = hq;
        case PMC
            hp_prev = -hp;
            hq_prev = -hq;
        case Bloch
            [e_prev h_prev] = fetch_eh2d(file_base_name, normal, Nr, gi, [false false false; true true true]);
            hp_prev = h_prev{Pp}.array / gi.exp_neg_ikL(Rr);
            hq_prev = h_prev{Qq}.array / gi.exp_neg_ikL(Rr);
        otherwise
            error('Not a supported boundary condition');
    end
end
hp = (hp + hp_prev) / 2;
hq = (hq + hq_prev) / 2;

clear hp_prev hq_prev
    
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

clear ep eq hp hq

gk(Pp) = Dual;
gk(Qq) = Dual;
gk(Rr) = Prim;
sr = scalar2d(strcat('S',AxisName(Rr)), sr_array, normal, intercept, gk, gi, true);
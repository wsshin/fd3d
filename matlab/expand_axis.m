function NaxisByNsign = expand_axis(val)

const;

if isvector(val)
    if length(val) == 1
        NaxisByNsign = [val val; val val; val val];
    else
        assert(length(val) == Naxis);
        NaxisByNsign = [val(Xx) val(Xx); val(Yy) val(Yy); val(Zz) val(Zz)];
    end
else
    assert(isequal(size(val),[Naxis Nsign]))
    NaxisByNsign = val;
end
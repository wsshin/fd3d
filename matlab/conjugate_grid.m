function conj_grid = conjugate_grid(grid_kind)

const;
if grid_kind == Prim
    conj_grid = Dual;
else
    assert(grid_kind==Dual)
    conj_grid = Prim;
end
function [e h] = fetch_eh(file_base_name, gi, comp_to_get)

const; 

if nargin <= 2
    comp_to_get = [true true true; true true true];
end

%% Read E field.
E = [];
if any(comp_to_get(Efield,:))
    if exist(strcat(file_base_name, '.sp.E')', 'file')==2
        E = read_solution(strcat(file_base_name, '.sp.E'));
    else
        E = read_solution(strcat(file_base_name, '.E'));
    end
end
e = create_e(E, gi, comp_to_get(Efield,:));
clear E

%% Read H field.
H = [];
if any(comp_to_get(Hfield,:))
    if exist(strcat(file_base_name, '.sp.H')', 'file')==2
        H = read_solution(strcat(file_base_name, '.sp.H'));
    else
        H = read_solution(strcat(file_base_name, '.H'));
    end
end
h = create_h(H, gi, comp_to_get(Hfield,:));
clear H;

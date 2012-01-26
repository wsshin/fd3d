function s = s_param(depth, N_PML, dx, omega)
if N_PML == 0
    s = 1;
else
    param_flag = 2;
    
    switch param_flag
        case 1
            logR = -16;
            m = 3.7;
        case 2
            logR = -1000;
            m=7;
        case 3
            logR = -1e8;
            m=10.9;
    end
    
    kappa_max = 1;

    sigma_max = -(m+1) * logR/(2*N_PML * dx);  % -(m+1) log(R)/(2 eta d), where eta = 1 and d = N_PML dx_orig/(1/k_0) = N_PML dx for the normalized equations
    sigma = sigma_max * (depth/N_PML)^m;
    kappa = 1 + (kappa_max-1) * (depth/N_PML)^m;

    ma = m;
    amax = 0;
    a = amax * (1-depth/N_PML)^ma;

    s = kappa + 1 * sigma/(a+sqrt(-1)*(omega));  % s = kappa + sigma_x/(i omega eps), but omega = 1 and eps = 1 when normalized.s = kappa + eta0*sigma/sqrt(-1);
end
function dydt = trueODEfunc_multiguild(~, y, p)
    % State variables
    H2 = y(1); CO2 = y(2); CH4 = y(3); H2S = y(4);
    SO4 = y(5); FeS = y(6); X_meth = y(7); X_sulf = y(8); X_aceto = y(9); Acetate = y(10);
    
    % Parameters
    k_meth = p(1); k_sulf = p(2); k_aceto = p(3);
    Y_m = p(4); Y_s = p(5); Y_a = p(6);
    KI_meth = p(7); KI_sulf = p(8); KI_aceto = p(9);
    k_precip = p(10); H2S_sat = p(11); H2_thresh = p(12);
    DG_thresh = p(13);

    % Constants
    R = 8.314e-3; T = 298.15; RT = R*T;
    DG0_meth = -130; DG0_sulf = -152; DG0_aceto = -95;

    % Inhibition terms
    f_inh_meth  = KI_meth  / (KI_meth  + H2S);
    f_inh_sulf  = KI_sulf  / (KI_sulf  + H2S);
    f_inh_aceto = KI_aceto / (KI_aceto + H2S);
    f_activation = H2 / (H2 + H2_thresh);

    % Safety for logs
    H2 = max(H2,1e-6); CO2 = max(CO2,1e-6); CH4 = max(CH4,1e-6);
    SO4 = max(SO4,1e-6); H2S = max(H2S,1e-6); Acetate = max(Acetate,1e-6);

    % Reaction quotients & Thermodynamics
    Q_meth  = CH4     / (H2^4 * CO2);
    Q_sulf  = H2S     / (H2^4 * SO4);
    Q_aceto = Acetate / (H2^4 * CO2^2);

    DG_meth  = DG0_meth  + RT*log(Q_meth);
    DG_sulf  = DG0_sulf  + RT*log(Q_sulf);
    DG_aceto = DG0_aceto + RT*log(Q_aceto);

    f_thermo_meth  = 1 / (1 + exp((DG_meth  - DG_thresh)/RT));
    f_thermo_sulf  = 1 / (1 + exp((DG_sulf  - DG_thresh)/RT));
    f_thermo_aceto = 1 / (1 + exp((DG_aceto - DG_thresh)/RT));

    % Rates
    r_meth  = k_meth  * H2 * CO2^(-2) * f_inh_meth  * f_activation * f_thermo_meth;
    r_sulf  = k_sulf  * H2 * SO4      * f_inh_sulf  * f_activation * f_thermo_sulf;
    r_aceto = k_aceto * H2 * CO2^2    * f_inh_aceto * f_activation * f_thermo_aceto;
    r_precip = k_precip * max(0, H2S - H2S_sat);

    % Derivatives
    dH2      = -4*r_meth - 4*r_sulf - 4*r_aceto;
    dCO2     = -1*r_meth - 2*r_aceto;
    dCH4     = +1*r_meth;
    dH2S     = +1*r_sulf - r_precip;
    dSO4     = -1*r_sulf;
    dFeS     = +1*r_precip;
    dX_meth  = Y_m * r_meth;
    dX_sulf  = Y_s * r_sulf;
    dX_aceto = Y_a * r_aceto;
    dAcetate = +1*r_aceto;

    dydt = [dH2; dCO2; dCH4; dH2S; dSO4; dFeS; dX_meth; dX_sulf; dX_aceto; dAcetate];
end
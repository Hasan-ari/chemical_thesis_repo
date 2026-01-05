
function anaerobic_model_two_phase_mixedSR_v3
    % ------------------ EXPERIMENT SETTINGS ------------------
    Vg = 0.14;   % headspace volume [L] ~ 140 mL
    Vl = 0.015;  % liquid volume [L]    ~ 15 mL
    T  = 298.15; % K (25°C)

    % Henry constants (Hcp) in mmol/L/atm (25°C; tune if needed)
    Hcp_H2   = 0.78;    % H2
    Hcp_CO2  = 33.0;    % CO2
    Hcp_H2S  = 90.0;    % H2S (molecular)

    % ------------------ Load experimental data --------------------
    raw   = readmatrix('Muller_2024_H2_Sandstone_at_25C.txt');
    t_exp = raw(:,1);                          % days
    nH2_g_exp   = raw(:,2) / 1000;             % µmol -> mmol
    nCO2_g_exp  = raw(:,3) / 1000;             % µmol -> mmol
    nCH4_g_exp  = raw(:,4) / 1000;             % µmol -> mmol
    nH2S_g_exp  = raw(:,5) / 1000;             % µmol -> mmol
    pH_exp      = raw(:,6);                    % pH
    SO4_exp     = raw(:,7);                    % mM (mmol/L)

    data_exp = [nH2_g_exp, nCO2_g_exp, nCH4_g_exp, nH2S_g_exp, SO4_exp];

    % pH interpolant
    pH_fun = @(t) max(0, interp1(t_exp, pH_exp, t, 'linear', 'extrap'));

    % ------------------ Initial aqueous equilibrium ----------------
    R_gas = 0.082057;  % L·atm/(mol·K)
    pH2_0   = (nH2_g_exp(1)/1000)  * R_gas * T / Vg; % atm
    pCO2_0  = (nCO2_g_exp(1)/1000) * R_gas * T / Vg; % atm
    H2_aq0  = Hcp_H2  * pH2_0;     % mmol/L
    CO2_aq0 = Hcp_CO2 * pCO2_0;    % mmol/L

    % ------------------ State vector (13 species) ------------------
    % [ nH2_g, nCO2_g, nCH4_g, nH2S_g, H2_aq, CO2_aq, SO4, FeS, X,
    %   Acetate, HCO3, S_aq_tot, Lag ]
    % S_aq_tot = H2S(aq)+HS- ; Lag is a smooth startup factor
    y0 = [ nH2_g_exp(1), nCO2_g_exp(1), nCH4_g_exp(1), nH2S_g_exp(1), ...
           H2_aq0, CO2_aq0, SO4_exp(1), 0.01, 0.01, 0, 0, 0, 0 ];

    % ------------------ Parameters to fit --------------------------
    % p = [k_meth, k_sulf, k_aceto,          ... vmax (1/day), multiply by X
    %      Y_m, Y_s, Y_Xa,                  ... biomass yields
    %      KI_H2S_meth, KI_H2S_sulf, KI_H2S_aceto,
    %      k_precip, HS_sat, H2_thresh, DG_thresh,
    %      K_H2, K_SO4, K_CO2,              ... Monod half-sat. (mM)
    %      kla_H2, kla_CO2, kla_H2S,        ... 1/day
    %      b_decay,                         ... biomass decay (1/day)
    %      t_lag, w_lag]                    ... smooth lag (days)
    p0 = [ 0.06,  0.08,  0.03,  ...
           0.06,  0.05,  0.05,  ...
           0.2,   0.2,   0.2,   ...
           0.02,  0.10,  0.02, -12, ...
           0.5,   0.5,   0.8,  ...
           10,    10,    8,    ...
           0.01,  3.0,   0.7];

    lb = [ 1e-4, 1e-4,  1e-4, ...
           0.01, 0.01, 0.01, ...
           1e-3, 1e-3, 1e-3, ...
           0,    0,    0,   -50, ...
           1e-3, 1e-3, 1e-3, ...
           0.1,  0.1,  0.1, ...
           0,    0,    0.1];

    ub = [ 5,    5,     5,   ...
           0.5,  0.5,   0.5, ...
           5,    5,     5,   ...
           1,    5,     1,    0, ...
           20,   20,    20, ...
           200,  200,   200, ...
           0.2,  10,     2.0];

    fit_opts = optimoptions('lsqnonlin','Display','iter','MaxFunctionEvaluations',6000);

    env.Vg = Vg; env.Vl = Vl; env.T = T; env.Rgas = R_gas;
    env.Hcp_H2 = Hcp_H2; env.Hcp_CO2 = Hcp_CO2; env.Hcp_H2S = Hcp_H2S;
    env.pH_fun  = pH_fun; env.pKa_H2S = 7.05;  % 25°C

    [p_fit,~,~,~,~,~,~] = lsqnonlin(@(p) residuals_mixed(p, t_exp, data_exp, y0, env), p0, lb, ub, fit_opts);
save('best_fit_params.mat','p_fit','env','y0');
fprintf('Best-fit parameters saved to best_fit_params.mat\n');
    % ------------------ Simulate with fitted parameters ------------
    odes = @(t,y) model_mixed(t,y,p_fit,env);
    opts = odeset('NonNegative',1:13,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
    [t_sim, y_sim] = ode15s(odes, [0 t_exp(end)], y0, opts);

    % Speciate sulfide for plotting
    [H2S_aq, HS_aq] = speciate_sulfide(y_sim(:,12), env.pH_fun(t_sim), env.pKa_H2S);

    % ------------------ Write results ------------------------------
    fileID = fopen('H2_Sandstone_25C_mixedSR.dat','w');
    fprintf(fileID, 'Time(days) nH2_g nCO2_g nCH4_g nH2S_g H2_aq CO2_aq SO4 FeS X Acetate HCO3 S_aq_tot H2S_aq HS Lag\n');
    fmt = '%10.6f %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g %12.6g\n';
    for i = 1:length(t_sim)
        fprintf(fileID, fmt, t_sim(i), y_sim(i,:), H2S_aq(i), HS_aq(i), y_sim(i,13));
    end
    fclose(fileID);

    % ------------------ Plots ------------------------------
    species = {'nH2_g','nCO2_g','nCH4_g','nH2S_g','H2(aq)','CO2(aq)','SO4','FeS','Biomass','Acetate','HCO3','S_{aq}^{tot}'};
    figure;
    for i = 1:length(species)
        subplot(6,2,i)
        if i <= 4
            plot(t_exp, data_exp(:,i), 'ko', 'DisplayName','Exp'); hold on;
            plot(t_sim, y_sim(:,i), 'b-', 'DisplayName','Model');
            ylabel('mmol (gas)'); xlabel('days');
        elseif i == 7
            plot(t_exp, data_exp(:,5), 'ko', 'DisplayName','Exp'); hold on;
            plot(t_sim, y_sim(:,7), 'b-', 'DisplayName','Model'); ylabel('mmol/L'); xlabel('days');
        else
            plot(t_sim, y_sim(:,i), 'b-', 'DisplayName','Model'); ylabel('mmol/L'); xlabel('days');
        end
        title(species{i}); legend;
    end

    figure;
    subplot(3,1,1); plot(t_sim, H2S_aq, 'b-'); title('H2S(aq)'); ylabel('mmol/L'); xlabel('days');
    subplot(3,1,2); plot(t_sim, HS_aq,  'b-'); title('HS^- (aq)'); ylabel('mmol/L'); xlabel('days');
    subplot(3,1,3); plot(t_sim, env.pH_fun(t_sim), 'k-'); title('pH(t)'); ylabel('pH'); xlabel('days');

    % Rates
    rates = zeros(length(t_sim), 4);
    for i = 1:length(t_sim)
        rates(i,:) = rate_out_mixed(t_sim(i), y_sim(i,:), p_fit, env);
    end
    figure;
    plot(t_sim, rates(:,1), 'r-', 'DisplayName','r_{meth}'); hold on;
    plot(t_sim, rates(:,2), 'b-', 'DisplayName','r_{sulf}');
    plot(t_sim, rates(:,3), 'g-', 'DisplayName','r_{precip}');
    plot(t_sim, rates(:,4), 'm-', 'DisplayName','r_{aceto}');
    title('Rates (mmol/L/day)'); xlabel('days'); ylabel('mmol/L/day'); legend;
end

%---------- Residuals ----------
function res = residuals_mixed(p, t_exp, data_exp, y0, env)
    try
        odes = @(t,y) model_mixed(t,y,p,env);
        opts = odeset('NonNegative',1:13,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
        [~, y_sim] = ode15s(odes, t_exp, y0, opts);
        sim_mat = [y_sim(:,1), y_sim(:,2), y_sim(:,3), y_sim(:,4), y_sim(:,7)];
        log_sim = log1p(sim_mat);
        log_exp = log1p([data_exp(:,1), data_exp(:,2), data_exp(:,3), data_exp(:,4), data_exp(:,5)]);
        weights = [1, 1, 0.9, 1.0, 1.1]; % a bit more weight on nH2S_g, SO4
        res = (log_sim - log_exp) .* weights;
        if any(y_sim(:) < -1e-9), res = res + 1e3 * abs(min(y_sim(:))); end
        res = res(:);
    catch
        res = 1e6 * ones(numel(data_exp), 1);
    end
end


% ---------- Model (ODEs) ----------
function dydt = model_mixed(t, y, p, env)
    Vg=env.Vg; Vl=env.Vl; T=env.T; Rgas=env.Rgas;
    Hcp_H2=env.Hcp_H2; Hcp_CO2=env.Hcp_CO2; Hcp_H2S=env.Hcp_H2S;
    pH=env.pH_fun(t); pKa=env.pKa_H2S;

    % State
    nH2_g=y(1); nCO2_g=y(2); nCH4_g=y(3); nH2S_g=y(4);
    H2_aq=y(5); CO2_aq=y(6); SO4=y(7); FeS=y(8); X=y(9);
    Ac=y(10); HCO3=y(11); S_tot=y(12); Lag=y(13);

    % Params
    k_m=p(1); k_s=p(2); k_a=p(3);
    Y_m=p(4); Y_s=p(5); Y_a=p(6);
    KI_m=p(7); KI_s=p(8); KI_a=p(9);
    k_prec=p(10); HS_sat=p(11); H2_th=p(12); DG_th=p(13);
    K_H2=p(14); K_SO4=p(15); K_CO2=p(16);
    kla_H2=p(17); kla_CO2=p(18); kla_H2S=p(19);
    b=p(20); t_lag=p(21); w_lag=p(22);

    % Thermo constants
    RkJ=8.314e-3; RT=RkJ*T;
    DG0_m=-130; DG0_s=-152; DG0_a=-95;

    % Guards
    eps=1e-12;
    nH2_g=max(nH2_g,eps); nCO2_g=max(nCO2_g,eps); nCH4_g=max(nCH4_g,eps); nH2S_g=max(nH2S_g,eps);
    H2_aq=max(H2_aq,eps); CO2_aq=max(CO2_aq,eps); SO4=max(SO4,eps); S_tot=max(S_tot,eps);
    Ac=max(Ac,eps); HCO3=max(HCO3,eps); X=max(X,eps);

    % Gas partial pressures
    pH2=(nH2_g/1000)*Rgas*T/Vg; pCO2=(nCO2_g/1000)*Rgas*T/Vg; pH2S=(nH2S_g/1000)*Rgas*T/Vg;

    % Henry equilibria
    Ceq_H2=Hcp_H2*pH2; Ceq_CO2=Hcp_CO2*pCO2; Ceq_H2S=Hcp_H2S*pH2S;

    % Transfers (mmol/L/day)
    J_H2 = kla_H2 *(Ceq_H2  - H2_aq);
    J_CO2= kla_CO2*(Ceq_CO2 - CO2_aq);

    % Speciation in liquid
    frac_HS=1/(1+10^(pKa-pH)); frac_H2S=1-frac_HS;
    HS_aq=S_tot*frac_HS; H2S_aq=S_tot*frac_H2S;

    % Degassing of molecular H2S
    J_H2S = kla_H2S*(Ceq_H2S - H2S_aq);

    % Inhibition by HS-
    f_inh_m = KI_m/(KI_m+HS_aq);
    f_inh_s = KI_s/(KI_s+HS_aq);
    f_inh_a = KI_a/(KI_a+HS_aq);

    % Activation: dissolved H2 + smooth lag gate
    f_H2 = H2_aq/(H2_aq+H2_th);
    f_lag = 1/(1+exp((t_lag - t)/max(w_lag,1e-3))); % 0->1 sigmoid around t_lag
    f_act = f_H2 * f_lag;

    % Monod saturations
    mH2  = H2_aq/(K_H2 + H2_aq);
    mSO4 = SO4  /(K_SO4+ SO4);
    mCO2 = CO2_aq/(K_CO2+ CO2_aq);

    % Thermo quotients
    Q_s = (H2S_aq^0.93) * (HS_aq^0.07) * (HCO3^1.93) / (H2_aq^4 * SO4 * CO2_aq^1.93);
    Q_m = 1;  % keep mild gating for methanogenesis to avoid over-constraining
    Q_a = Ac / (H2_aq^4 * CO2_aq^2);

    DG_s = DG0_s + RT*log(Q_s);
    DG_m = DG0_m;                 % mild gate
    DG_a = DG0_a + RT*log(Q_a);

    fT_s = 1/(1+exp((DG_s - DG_th)/RT));
    fT_m = 1/(1+exp((DG_m - DG_th)/RT));
    fT_a = 1/(1+exp((DG_a - DG_th)/RT));

    % Rates (mmol/L/day) -- biomass mediated
    r_meth  = k_m * X * mH2 * mCO2           * f_inh_m * f_act * fT_m;
    r_sulf  = k_s * X * mH2 * mSO4 * mCO2    * f_inh_s * f_act * fT_s;
    r_aceto = k_a * X * mH2 * (mCO2.^2)      * f_inh_a * f_act * fT_a;

    % Precipitation (from HS- only)
    r_prec = k_prec * max(0, HS_aq - HS_sat);

    % Gas balances (mmol/day)
    dnH2_g  = - J_H2  * Vl;
    dnCO2_g = - J_CO2 * Vl;
    dnCH4_g = + r_meth * Vl;     % CH4 to gas
    dnH2S_g = + J_H2S * Vl;      % degassing of H2S(aq)

    % Liquid balances (mmol/L/day)
    dH2_aq   = + J_H2  - 4*r_meth - 4*r_sulf - 4*r_aceto;
    dCO2_aq  = + J_CO2 - 1*r_meth - 2*r_aceto - 1.93*r_sulf;  % mixed SR consumes CO2
    dSO4     = - 1*r_sulf;
    dFeS     = + r_prec;
    dX       = + Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X; % biomass growth & decay
    dAc      = + r_aceto;
    dHCO3    = + 1.93*r_sulf;    % from mixed SR reaction
    dS_tot   = + 1.00*r_sulf - r_prec - J_H2S; % all sulfide born in liquid, then precip/degass
    dLag     = (f_lag - Lag)/max(w_lag,1e-3); % optional: track smooth lag state (for plotting)

    dydt = [dnH2_g; dnCO2_g; dnCH4_g; dnH2S_g; dH2_aq; dCO2_aq; dSO4; dFeS; dX; dAc; dHCO3; dS_tot; dLag];
end

% ---------- Speciation helper ----------
function [H2S_aq, HS_aq] = speciate_sulfide(S_tot, pH, pKa)
    frac_HS  = 1 ./ (1 + 10.^(pKa - pH));
    HS_aq  = S_tot .* frac_HS;
    H2S_aq = S_tot - HS_aq;
end

% ---------- Rates for plotting ----------
function dr = rate_out_mixed(t, y, p, env)
    H2_aq=max(y(5),1e-12); CO2_aq=max(y(6),1e-12); SO4=max(y(7),1e-12); X=max(y(9),1e-12); S_tot=max(y(12),1e-12);
    pH=env.pH_fun(t); frac_HS=1/(1+10^(env.pKa_H2S - pH)); HS=S_tot*frac_HS;

    k_m=p(1); k_s=p(2); k_a=p(3);
    KI_m=p(7); KI_s=p(8); KI_a=p(9);
    H2_th=p(12); K_H2=p(14); K_SO4=p(15); K_CO2=p(16);
    t_lag=p(21); w_lag=p(22);

    f_inh_m = KI_m/(KI_m+HS); f_inh_s = KI_s/(KI_s+HS); f_inh_a = KI_a/(KI_a+HS);
    f_H2 = H2_aq/(H2_aq+H2_th); f_lag = 1/(1+exp((t_lag - t)/max(w_lag,1e-3))); f_act=f_H2*f_lag;

    mH2=H2_aq/(K_H2+H2_aq); mSO4=SO4/(K_SO4+SO4); mCO2=CO2_aq/(K_CO2+CO2_aq);
    r_meth  = k_m * X * mH2 * mCO2          * f_inh_m * f_act;
    r_sulf  = k_s * X * mH2 * mSO4 * mCO2   * f_inh_s * f_act;
    r_aceto = k_a * X * mH2 * (mCO2.^2)     * f_inh_a * f_act;
    r_prec=0;
    dr=[r_meth, r_sulf, r_prec, r_aceto];
end

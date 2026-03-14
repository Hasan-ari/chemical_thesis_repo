
function anaerobic_model_two_phase_mixedSR_25C_v4
% Gypsum @ 25 °C with effective Henry scale factors and H2S flux patch
% - T = 25 °C (298.15 K)
% - Hscp (c/p) in mmol/L/atm; partial pressures in atm
% - Minimal H2S patch: define outgassing-positive flux and use it in gas & liquid
% - Writes reaction rates to .dat; plots species and rates
% -------------------- EXPERIMENT SETTINGS --------------------
Vg = 0.14; % headspace volume [L] ~ 140 mL
Vl = 0.015; % liquid volume [L] ~ 15 mL
T = 298.15; % K (25°C)
R_gas = 0.082057; % L·atm/(mol·K)

% Henry solubility constants Hscp (c/p) @ 25°C, mmol/L/atm
% adjust if needed
%%% NEW (Gypsum): use 25 °C base constants + scale factors (solubility variant Hscp=c/p)
Hcp_H2_base  = 0.78;  % H2 (25 °C) (mmol/L/atm) [1](https://www.webqc.org/henry_gas_law.html)
Hcp_CO2_base = 34.0;  % CO2 (25 °C) (mmol/L/atm) [1](https://www.webqc.org/henry_gas_law.html)
Hcp_H2S_base = 90.0;  % H2S (25 °C) (mmol/L/atm)

% Optional scale factors (keep 1.0 unless you need fine-tuning)
phi_H2  = 1.00;
phi_CO2 = 1.00;
% (You can add phi_H2S if you need it; I keep H2S as base here)

% Effective Henry constants used by the model
Hcp_H2_eff  = phi_H2  * Hcp_H2_base;
Hcp_CO2_eff = phi_CO2 * Hcp_CO2_base;
Hcp_H2S_eff = Hcp_H2S_base;

% -------------------- Load experimental data --------------------
raw = readmatrix('Muller_2024_H2_Gypsum_at_25C.txt');
t_exp       = raw(:,1);  % days
nH2_g_exp   = raw(:,2) / 1000; % µmol -> mmol
nCO2_g_exp  = raw(:,3) / 1000; % µmol -> mmol
nCH4_g_exp  = raw(:,4) / 1000; % µmol -> mmol
nH2S_g_exp  = raw(:,5) / 1000; % µmol -> mmol
pH_exp      = raw(:,6);        % pH
SO4_exp     = raw(:,7);        % sulfate mM (mmol/L)
data_exp    = [nH2_g_exp, nCO2_g_exp, nCH4_g_exp, nH2S_g_exp, SO4_exp];

% pH interpolant
pH_fun = @(t) max(0, interp1(t_exp, pH_exp, t, 'linear', 'extrap'));

% -------------------- Initial aqueous equilibrium --------------------
pH2  = (nH2_g_exp(1)/1000)  * R_gas * T / Vg; % atm
pCO2 = (nCO2_g_exp(1)/1000) * R_gas * T / Vg; % atm
pH2S = (nH2S_g_exp(1)/1000) * R_gas * T / Vg; % atm

H2_aq0  = Hcp_H2_eff  * pH2;   % mmol/L
CO2_aq0 = Hcp_CO2_eff * pCO2;  % mmol/L

% Diagnostics
Ptot_0 = ((nH2_g_exp(1)+nCO2_g_exp(1)+nCH4_g_exp(1)+nH2S_g_exp(1))/1000) * R_gas * T / Vg;
fprintf('\n[Henry/Pressure @ t0] P_tot=%.3f atm \n p_H2=%.3f, p_CO2=%.3f, p_H2S=%.4f atm \n Ceq: H2=%.4f, CO2=%.3f mmol/L\n', ...
        Ptot_0, pH2, pCO2, pH2S, H2_aq0, CO2_aq0);

%% -------------------- Initial states (13 species) --------------------
% [ nH2_g, nCO2_g, nCH4_g, nH2S_g, H2(aq), CO2(aq), SO4, FeS, X, Acetate, HCO3, S_tot, Lag ]
S_tot0 = 0.02; % %%% PATCH: tiny seed (0.02 mM) to allow early H2S(g) appearance (optional)
y0 = [ nH2_g_exp(1), nCO2_g_exp(1), nCH4_g_exp(1), nH2S_g_exp(1), ...
       H2_aq0, CO2_aq0, SO4_exp(1), ...
       0.01, 0.01, 0, 0, S_tot0, 0 ];

%% -------------------- Parameter vector (unchanged biology) --------------------
% -------------------- Parameters to fit --------------------
% p = [k_m, k_s, k_a, Y_m, Y_s, Y_a, KI_m, KI_s, KI_a, k_prec, HS_sat, H2_th, DG_th, ...
%      K_H2, K_SO4, K_CO2, kla_H2, kla_CO2, kla_H2S, b, t_lag, w_lag]
% %%% NEW (Gypsum): append two parameters for gypsum behaviour:
% ... k_diss_gyp (1/day), beta_SO4_m (mM^-1)

p0 = [ 0.06, 0.08, 0.03, ...
       0.06, 0.05, 0.05, ...
       0.20, 0.20, 0.20, ...
       0.02, 0.10, 0.02, -12, ...
       0.50, 0.50, 0.80, ...
       10.0, 10.0, 8.0, ...
       0.01, 3.0, 0.7, ...
       0.30, 0.10 ];

% %%% PATCH: disable FeS precipitation (no Fe pool tracked), increase H2S kla, tune gypsum dissolution
p0(10) = 0.0;   % k_prec -> off for gypsum fits
% p0(11) stays as-is (HS_sat), irrelevant with k_prec=0
p0(19) = 25.0;  % kla_H2S initial guess (was 8.0)
p0(23) = 0.12;  % k_diss_gyp initial guess (was 0.30), consistent with ~2-week equilibration

lb = [ 1e-4, 1e-4, 1e-4, ...
       0.01, 0.01, 0.01, ...
       1e-3, 1e-3, 1e-3, ...
       0, 0, 0, -50, ...
       1e-3, 1e-3, 1e-3, ...
       0.1, 0.1, 0.1, ...
       0, 0, 0.1, ...
       0.01, 0.00 ];

ub = [ 5, 5, 5, ...
       0.5, 0.5, 0.5, ...
       5, 5, 5, ...
       1, 5, 1, 0, ...
       20, 20, 20, ...
       200, 200, 200, ...
       0.2, 10, 2.0, ...
       2.00, 1.00 ];

fit_opts = optimoptions('lsqnonlin','Display','iter','MaxFunctionEvaluations',6000);

%% -------------------- Environment pack --------------------
env.Vg = Vg; env.Vl = Vl; env.T = T; env.Rgas = R_gas;
env.Hcp_H2_eff  = Hcp_H2_eff;
env.Hcp_CO2_eff = Hcp_CO2_eff;
env.Hcp_H2S_eff = Hcp_H2S_eff;
env.pH_fun      = pH_fun;
env.pKa_H2S     = 7.05; % modest T-adjusted pKa1; tweak only if needed

% %%% PATCH: gypsum-equilibrium SO4 at 25 °C ~30–40 mM (was 15.0 mM)
env.SO4_sat_gyp = 36.0; % mM — matches experimental plateaus (biotic ≈36 mM; abiotic ≈32 mM)

%% -------------------- Fit (solve residuals at t_exp) --------------------
[p_fit,~,~,~,~,~,~] = lsqnonlin(@(p) residuals_full(p, t_exp, data_exp, y0, env), p0, lb, ub, fit_opts);
save('best_fit_params_Gypsum_25C.mat','p_fit','env','y0');

%% -------------------- Final simulation (dense grid) --------------------
odes = @(t,y) model_mixed(t,y,p_fit,env);
opts = odeset('NonNegative',1:13,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
[t_sim, y_sim] = ode15s(odes, [0 t_exp(end)], y0, opts);

% Sulfide speciation
[H2S_aq, HS_aq] = speciate_sulfide(y_sim(:,12), env.pH_fun(t_sim), env.pKa_H2S);

% Reaction rates over t_sim
rates = zeros(length(t_sim), 4); % [r_meth, r_sulf, r_precip, r_aceto]
for k = 1:length(t_sim)
    rates(k,:) = rate_out_mixed(t_sim(k), y_sim(k,:), p_fit, env);
end

%% -------------------- Write results (.dat) INCLUDING reaction rates --------------------
fileID = fopen('Gypsum_25C_inc_rates.dat','w');
fprintf(fileID, ['Time(days) nH2_g nCO2_g nCH4_g nH2S_g H2_aq CO2_aq SO4 FeS X Acetate HCO3 ', ...
                 'S_tot H2S_aq HS Lag r_meth r_sulf r_precip r_aceto\n']);
fmt = ['%10.6f ', ...
       '%12.6g %12.6g %12.6g %12.6g ', ...
       '%12.6g %12.6g %12.6g %12.6g ', ...
       '%12.6g %12.6g %12.6g %12.6g ', ...
       '%12.6g %12.6g %12.6g ', ...
       '%12.6g %12.6g %12.6g %12.6g\n'];
for i = 1:length(t_sim)
    fprintf(fileID, fmt, ...
        t_sim(i), ...
        y_sim(i,1), y_sim(i,2), y_sim(i,3), y_sim(i,4), ...
        y_sim(i,5), y_sim(i,6), y_sim(i,7), y_sim(i,8), ...
        y_sim(i,9), y_sim(i,10), y_sim(i,11), y_sim(i,12), ...
        H2S_aq(i), HS_aq(i), y_sim(i,13), ...
        rates(i,1), rates(i,2), rates(i,3), rates(i,4));
end
fclose(fileID);
fprintf('Wrote Gypsum_25C_inc_rates.dat (incl. reaction rates)\n');

% -------------------- Plots --------------------
%% Figure 1: Gases & Aqueous - Gypsum (25 °C)
species = {'nH2_g','nCO2_g','nCH4_g','nH2S_g','H2(aq)','CO2(aq)','SO4','FeS','Biomass','Acetate','HCO3','S_{aq}^{tot}'};
figure('Name','Gases & Aqueous - Gypsum (25 °C)');
for i = 1:length(species)
    subplot(6,2,i)
    if i <= 4
        plot(t_exp, data_exp(:,i), 'ko', 'DisplayName','Exp'); hold on;
        plot(t_sim, y_sim(:,i), 'b-', 'DisplayName','Model');
        ylabel('mmol (gas)'); xlabel('days');
    elseif i == 7
        plot(t_exp, data_exp(:,5), 'ko', 'DisplayName','SO4 Exp'); hold on;
        plot(t_sim, y_sim(:,7), 'b-', 'DisplayName','SO4 Model'); ylabel('mmol/L'); xlabel('days');
    else
        plot(t_sim, y_sim(:,i), 'b-', 'DisplayName','Model'); ylabel('mmol/L'); xlabel('days');
    end
    title(species{i}); legend;
end

%% Figure 2: Sulfide speciation & pH
figure('Name','Sulfide speciation & pH');
subplot(3,1,1); plot(t_sim, H2S_aq, 'b-'); title('H2S(aq)'); ylabel('mmol/L'); xlabel('days');
subplot(3,1,2); plot(t_sim, HS_aq,  'b-'); title('HS^- (aq)'); ylabel('mmol/L'); xlabel('days');
subplot(3,1,3); plot(t_sim, env.pH_fun(t_sim), 'k-'); title('pH(t)'); ylabel('pH'); xlabel('days');

% Rates
rates = zeros(length(t_sim), 4);
for i = 1:length(t_sim)
    rates(i,:) = rate_out_mixed(t_sim(i), y_sim(i,:), p_fit, env);
end

%% Figure 3: Kinetic rates (mmol/L/day)
figure('Name','Rates (mmol/L/day)');
plot(t_sim, rates(:,1), 'r-', 'DisplayName','r_{meth}'); hold on;
plot(t_sim, rates(:,2), 'b-', 'DisplayName','r_{sulf}');
plot(t_sim, rates(:,3), 'g-', 'DisplayName','r_{precip}');
plot(t_sim, rates(:,4), 'm-', 'DisplayName','r_{aceto}');
xlabel('days'); ylabel('mmol/L/day'); title('Kinetic Rates'); legend; legend;

%% RMSE at t_exp (interpolate model)
yH2_on_exp  = interp1(t_sim, y_sim(:,1), t_exp, 'linear', 'extrap');
yCO2_on_exp = interp1(t_sim, y_sim(:,2), t_exp, 'linear', 'extrap');
fprintf('RMSE (gas moles on t_exp): H2=%.4f mmol CO2=%.4f mmol\n', ...
        rmse_equal(yH2_on_exp, data_exp(:,1)), rmse_equal(yCO2_on_exp, data_exp(:,2)));
end % ---- end main ----

%% -------------------- Residuals (solve at t_exp) --------------------
function res = residuals_full(p, t_exp, data_exp, y0, env)
odes = @(t,y) model_mixed(t,y,p,env);
opts = odeset('NonNegative',1:13,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
[~, y_sim] = ode15s(odes, t_exp, y0, opts);

% Compare model to: nH2_g, nCO2_g, nCH4_g, nH2S_g, SO4
sim_mat = [y_sim(:,1), y_sim(:,2), y_sim(:,3), y_sim(:,4), y_sim(:,7)];
log_sim = log1p(sim_mat);
log_exp = log1p([data_exp(:,1), data_exp(:,2), data_exp(:,3), data_exp(:,4), data_exp(:,5)]);

% %%% PATCH: emphasize SO4 plateau more strongly (was 1.1)
weights = [1, 1, 0.9, 1.0, 2.0];

res = (log_sim - log_exp) .* weights;

% Elimination for negative states
if any(y_sim(:) < -1e-9), res = res + 1e3 * abs(min(y_sim(:))); end
res = res(:);
end

%% -------------------- ODE model (with H2S outgassing-positive flux) --------------------
function dydt = model_mixed(t, y, p, env)
Vg=env.Vg; Vl=env.Vl; T=env.T; Rgas=env.Rgas;
Hcp_H2=env.Hcp_H2_eff; Hcp_CO2=env.Hcp_CO2_eff; Hcp_H2S=env.Hcp_H2S_eff;
pH=env.pH_fun(t); pKa=env.pKa_H2S;

% State
nH2_g=y(1); nCO2_g=y(2); nCH4_g=y(3); nH2S_g=y(4);
H2_aq=y(5); CO2_aq=y(6); SO4=y(7); FeS=y(8); X=y(9);
Ac=y(10); HCO3=y(11); S_tot=y(12); Lag=y(13);

% Parameters
k_m  = p(1);  k_s  = p(2);  k_a  = p(3);
Y_m  = p(4);  Y_s  = p(5);  Y_a  = p(6);
KI_m = p(7);  KI_s = p(8);  KI_a = p(9);
k_prec = p(10); % biomass yield
HS_sat = p(11); H2_th = p(12); DG_th = p(13);
K_H2 = p(14); K_SO4 = p(15); K_CO2 = p(16);
kla_H2 = p(17); kla_CO2 = p(18); kla_H2S = p(19);
b = p(20); t_lag = p(21); w_lag = p(22);
% %%% NEW (Gypsum) indices:
k_diss_gyp = p(23); % 1/day
beta_SO4_m = p(24); % mM^-1

% Thermodynamics
% Constants -- Dynamic ΔG via Nernst Equation
RkJ=8.314e-3; RT=RkJ*T;
DG0_m=-130; DG0_s=-152; DG0_a=-95;

% Guards
eps=1e-12;
nH2_g=max(nH2_g,eps); nCO2_g=max(nCO2_g,eps); nCH4_g=max(nCH4_g,eps); nH2S_g=max(nH2S_g,eps);
H2_aq=max(H2_aq,eps); CO2_aq=max(CO2_aq,eps); SO4=max(SO4,eps); S_tot=max(S_tot,eps);
Ac=max(Ac,eps); HCO3=max(HCO3,eps); X=max(X,eps);

% Partial pressures (atm) from moles (mmol)
pH2  = (nH2_g /1000)  * Rgas * T / Vg;
pCO2 = (nCO2_g/1000)  * Rgas * T / Vg;
pH2S = (nH2S_g/1000)  * Rgas * T / Vg;

% Henry equilibria (mmol/L) @ 25 °C
Ceq_H2  = Hcp_H2  * pH2;
Ceq_CO2 = Hcp_CO2 * pCO2;
Ceq_H2S = Hcp_H2S * pH2S;

% Gas–liquid transfers for H2, CO2 (liquid-side uptake positive)
J_H2  = kla_H2  * (Ceq_H2  - H2_aq);
J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq);

% Sulfide speciation
frac_HS  = 1/(1+10^(pKa - pH));
frac_H2S = 1 - frac_HS;
HS_aq  = S_tot*frac_HS;
H2S_aq = S_tot*frac_H2S;

% H2S: outgassing-positive flux
Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S);

% Inhibitions & Activation: dissolved H2 + smooth lag gate
f_inh_m = KI_m/(KI_m+HS_aq);
f_inh_s = KI_s/(KI_s+HS_aq);
f_inh_a = KI_a/(KI_a+HS_aq);
f_H2    = H2_aq/(H2_aq+H2_th);
f_lag   = 1/(1+exp((t_lag - t)/max(w_lag,1e-3)));
f_act   = f_H2 * f_lag;

% Monod saturations
mH2  = H2_aq /(K_H2  + H2_aq);
mSO4 = SO4   /(K_SO4 + SO4);
mCO2 = CO2_aq/(K_CO2 + CO2_aq);

% Thermo gates (heuristics consistent with v3)
% Reaction quotients
% Q_s = (H2S_aq^0.93) * (HS_aq^0.07) * (HCO3^1.93) / (H2_aq^4 * SO4 * CO2_aq^1.93);
%%% FIX (Gypsum SR): SR Q_s should not depend on CO2/HCO3 for H2-driven SR
Q_s = 1; % keep SR thermo gate mild/neutral for gypsum
Q_a = Ac / (H2_aq^4 * CO2_aq^2);

% Dynamic Gibbs energies
DG_s = DG0_s + RT*log(Q_s);
DG_m = DG0_m; % mild gate
DG_a = DG0_a + RT*log(Q_a);

fT_s = 1/(1+exp((DG_s - DG_th)/RT));
fT_m = 1/(1+exp((DG_m - DG_th)/RT));
fT_a = 1/(1+exp((DG_a - DG_th)/RT));

%%% NEW (Gypsum): sulfate-vs-methanogen competition gate
f_comp_m = 1 / (1 + beta_SO4_m * SO4);

% Biomass-mediated rates (mmol/L/day)
r_meth = k_m * X * mH2 * mCO2 * f_inh_m * f_act * fT_m * f_comp_m; % <-- gate here
% %%% PATCH: REMOVE double application of competition gate
% r_meth = r_meth * f_comp_m; % (deleted)

% r_sulf = k_s * X * mH2 * mSO4 * mCO2 * f_inh_s * f_act * fT_s;
%%% FIX (Gypsum SR): remove CO2 gate from SR
r_sulf = k_s * X * mH2 * mSO4 * f_inh_s * f_act * fT_s;

r_aceto = k_a * X * mH2 * (mCO2.^2) * f_inh_a * f_act * fT_a;

% Precipitation (from HS-)
r_prec = k_prec * max(0, HS_aq - HS_sat); % %%% PATCH: k_prec=0 for gypsum fits

% -------------------- Gypsum dissolution source --------------------
% This captures that gypsum buffers sulfate (target ~30–40 mM at 25–40 °C) and SR keeps methanogenesis at bay while sulfate is present
SO4_sat = env.SO4_sat_gyp; % mM
r_diss_gyp = k_diss_gyp * max(0, SO4_sat - SO4); % mM/day

% -------------------- Gas balances (mmol/day) --------------------
dnH2_g  = - J_H2  * Vl;
dnCO2_g = - J_CO2 * Vl;
dnCH4_g = + r_meth * Vl;   % CH4 to gas
dnH2S_g = + Jout_H2S * Vl; % degassing of H2S(aq)

% -------------------- Differential equations --------------------
% Liquid balances (mmol/L/day)
dH2_aq  = + J_H2  - 4*r_meth - 4*r_sulf - 4*r_aceto;
% dCO2_aq = + J_CO2 - 1*r_meth - 2*r_aceto - 1.93*r_sulf; % mixed SR consumes CO2
%% FIX (Gypsum SR): SR does not consume CO2 in hydrogenotrophic SR
dCO2_aq = + J_CO2 - 1*r_meth - 2*r_aceto; % <-- remove -1.93*r_sulf

dSO4 = - 1*r_sulf + r_diss_gyp; %%% NEW (Gypsum): add dissolution source
dFeS = + r_prec;
dX   = + Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X; % biomass growth & decay
dAc  = + r_aceto;

% dHCO3 = + 1.93*r_sulf; % from mixed SR reaction
%% FIX (Gypsum SR): remove HCO3 production from SR
% hydrogenotrophic SR uses H2 and SO4^2-; gating by CO2 suppresses SR and shifts the whole system toward early methanogenesis
dHCO3 = 0 + 0*r_aceto; % keep zero (or add a separate carbonate speciation block later)

dS_tot = + 1.00*r_sulf - r_prec - Jout_H2S; % all sulfide born in liquid, then precip/degass aligned with outgassing-positive flux
dLag   = (f_lag - Lag)/max(w_lag,1e-3); % track smooth lag state (for plotting)

dydt = [dnH2_g; dnCO2_g; dnCH4_g; dnH2S_g; dH2_aq; dCO2_aq; dSO4; dFeS; dX; dAc; dHCO3; dS_tot; dLag];
end

%% -------------------- Reaction rates (for plotting + .dat) --------------------
function dr = rate_out_mixed(t, y, p, env)
H2_aq=max(y(5),1e-12); CO2_aq=max(y(6),1e-12); SO4=max(y(7),1e-12); X=max(y(9),1e-12); S_tot=max(y(12),1e-12);
pH=env.pH_fun(t); frac_HS=1/(1+10^(env.pKa_H2S - pH)); HS=S_tot*frac_HS;
k_m=p(1); k_s=p(2); k_a=p(3);
KI_m=p(7); KI_s=p(8); KI_a=p(9);
H2_th=p(12); K_H2=p(14); K_SO4=p(15); K_CO2=p(16);
t_lag=p(21); w_lag=p(22);

f_inh_m = KI_m/(KI_m+HS);
f_inh_s = KI_s/(KI_s+HS);
f_inh_a = KI_a/(KI_a+HS);
f_H2    = H2_aq/(H2_aq+H2_th);
f_lag   = 1/(1+exp((t_lag - t)/max(w_lag,1e-3)));
f_act   = f_H2 * f_lag;

mH2  = H2_aq /(K_H2  + H2_aq);
mSO4 = SO4   /(K_SO4 + SO4);
mCO2 = CO2_aq/(K_CO2 + CO2_aq);

r_meth = k_m * X * mH2 * mCO2 * f_inh_m * f_act;
% Competition gate (SR vs methanogenesis) -- (left commented in diagnostic output)
% f_comp_m = 1 / (1 + beta_SO4_m * SO4);
% r_meth = r_meth * f_comp_m;

% %%% PATCH: align SR rate with ODE (remove mCO2)
r_sulf = k_s * X * mH2 * mSO4 * f_inh_s * f_act;

r_aceto = k_a * X * mH2 * (mCO2.^2) * f_inh_a * f_act;

% keep precipitation zero in the rate file unless you want the exact gate:
r_prec = 0;
%If plan to use mineralogies with measurable Fe(II), 
% add a finite Fe_pool state and cap precipitation with
%  r_prec = k_prec * min(HS_excess, Fe_pool). That prevents over draining sulfide and lets you test iron limitation directly.

dr = [r_meth, r_sulf, r_prec, r_aceto];
end

%% -------------------- Speciation helper --------------------
function [H2S_aq, HS_aq] = speciate_sulfide(S_tot, pH, pKa)
frac_HS  = 1 ./ (1 + 10.^(pKa - pH));
HS_aq    = S_tot .* frac_HS;
H2S_aq   = S_tot - HS_aq;
end

%% -------------------- RMSE helper --------------------
function r = rmse_equal(a,b)
r = sqrt(mean((a(:)-b(:)).^2,'omitnan'));
end

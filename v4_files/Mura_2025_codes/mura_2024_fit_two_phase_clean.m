
function out = mura_2024_fit_two_phase_clean()
% Mura 2024 – Two-phase kinetic fit at ~60 bar, 36 °C (ϕ–γ framework)
% ----------------------------------------------------------------------
% This script/function reproduces your original model but clarifies all
% parameters, events, units, and roles. It:
%  1) Loads experimental time series (H2(g), CO2(g), SO4, formate, acetate, Ca).
%  2) Builds environment, initial conditions (PV/RT, CH4 cushion).
%  3) Defines a 41-parameter vector p with bounds and clear documentation.
%  4) Fits p via lsqnonlin on weighted log residuals.
%  5) Simulates with staged events (day7 gas makeup, day9 H2 injection, day17 water).
%  6) Reports plots, pH, rates, sulfur mass balance, and exports a .dat file.
%
% DATA SOURCE (columns):
% time_d, H2_g_mmol, CO2_g_mmol, SO4_mM, formate_mM, acetate_mM, calcium_mM
% (from: Mura2024_Fig1_readoff_with_calcium.txt)
%
% Experiment reference & protocol rationale:
% - Staged make-up and injections follow Mura et al. (2024) Front. Microbiol. 15:1439866
%   (~60 bar, 36 °C), day-7 CH4+CO2 reinjection, day-9 H2 pulse, day-17 water addition.
% - The initial headspace CO2 is set from first measurement; H2=0; CH4 balances PV/RT.

%% 0) Load experimental data
raw = readmatrix('Mura2024_Fig1_readoff_with_calcium.txt');  % time_d, H2g, CO2g, SO4, Formate, Acetate, Ca
t_exp = raw(:,1);
H2g_exp_mmol  = raw(:,2);
CO2g_exp_mmol = raw(:,3);
SO4_exp_mM    = raw(:,4);
Form_exp_mM   = raw(:,5);
Acet_exp_mM   = raw(:,6);
Ca_exp_mM     = raw(:,7);

data_exp = [H2g_exp_mmol, CO2g_exp_mmol, SO4_exp_mM, Form_exp_mM, Acet_exp_mM, Ca_exp_mM];

% Helper indices for first positive gas inventories
idxCO2_init = find(CO2g_exp_mmol > 0, 1, 'first');
if isempty(idxCO2_init), error('No positive CO2(g) value found.'); end
CO2g_init_mmol = CO2g_exp_mmol(idxCO2_init);

idxH2_init = find(H2g_exp_mmol > 0, 1, 'first');
if isempty(idxH2_init), idxH2_init = idxCO2_init; end
H2g_init_mmol = H2g_exp_mmol(idxH2_init);

%% 1) Environment & constants
env = struct();
env.T    = 309.15;   % K (36 °C)
env.Pset = 60.0;     % bar
env.Vg   = 0.25;     % L (initial gas volume)
env.Vl   = 0.25;     % L (initial liquid volume)

% Freshwater Henry constants Hcp (Hcp = c/p, mmol·L^-1·bar^-1) — baseline
env.Hcp_H2_fw  = 0.80;  % H2
env.Hcp_CO2_fw = 30.0;  % CO2
env.Hcp_H2S_fw = 85.0;  % H2S

% Calcium reference (for simple re-equilibration term)
env.Ca_eq      = max(Ca_exp_mM);

% Peng–Robinson EOS pure-component properties at T
env.PR         = getPRparams(env.T);  % cached a_i, b_i

% Sulfide acid/base constant (for speciation)
env.pKa_H2S    = 6.95;

% Staged protocol events (days)
env.t_events   = [7, 9, 17];  % day7 gas makeup, day9 H2 injection, day17 water addition
env.dVl_day17  = 0.353;       % L (added water on day 17)

% Initial CO2(g) set to first measurement; yCO2_set used only diagnostically
env.CO2g_init_mmol = CO2g_init_mmol;
yCO2_set = CO2g_init_mmol / max(CO2g_init_mmol + H2g_init_mmol, 1e-12);
env.yCO2_set = min(max(yCO2_set, 1e-4), 0.5);

% Smooth logistic pulses (avoid sharp jumps)
env.tau_pulse = 0.25;           % day (~6 hours window)
env.k_pulse   = 40/env.tau_pulse;

%% 2) Initial conditions (headspace via PV/RT; CH4 cushion)
Rbar = 0.08314472; % bar·L·(mol^-1·K^-1)
ntot_gas0_mol  = env.Pset * env.Vg/(Rbar*env.T);
ntot_gas0_mmol = 1000*ntot_gas0_mol;

nH2g0  = 0.0;                      % mmol
nCO2g0 = env.CO2g_init_mmol;      % from data (~30 mmol)
nH2Sg0 = 0.0;
nCH4g0 = max(ntot_gas0_mmol - nCO2g0 - nH2Sg0, 1.0);  % cushion gas
env.ntot_gas0_mmol = ntot_gas0_mmol;

% Aqueous seeds and carbonate initialization (CT0, Alk0 will be fitted)
H2_aq0 = 1e-6;     % mmol·L^-1
yCO2_0 = nCO2g0 / max(nH2g0+nCO2g0+nCH4g0+nH2Sg0, 1e-12);
CO2aq_guess = env.Hcp_CO2_fw*(yCO2_0*env.Pset); % mmol·L^-1
CT0  = max(CO2aq_guess,1e-6);  % mmol·L^-1
pH0  = 6.0;
S_tot0 = 1e-3;      % mmol·L^-1 (seed sulfide)
Alk0 = alk_from_CT_pH(CT0, pH0, S_tot0, env);

% Measured aqueous initializations
SO4_0   = max(SO4_exp_mM(1), 1e-6);
Form_0  = max(Form_exp_mM(1), 0);
Acet_0  = max(Acet_exp_mM(1), 0);
X0      = 0.12;        % biomass (arbitrary seed; fitted dynamics drive it)
Ca0     = max(Ca_exp_mM(1), 0);
FeS0    = 0.01;
Fe_pool0= 0.10;
Alk_0   = Alk0;

% State vector (15)
y0 = [nH2g0, nCO2g0, nCH4g0, nH2Sg0, H2_aq0, CT0, SO4_0, Form_0, Acet_0, ...
      S_tot0, X0, Ca0, FeS0, Fe_pool0, Alk_0]';

%% 3) Parameter vector p (TOTAL = 41) with documentation
% p(1..15): KINETICS & MONODS
%  k_m, k_s, k_a, k_f   : max specific rates for methanogenesis (m), sulfate red. (s),
%                         acetogenesis (a), formate production (f) [day^-1]
%  Y_m, Y_s, Y_a, Y_f   : biomass yields for each route [mmol_X per mmol_substrate]
%  KI_m, KI_s, KI_a     : sulfide inhibition half-constants [mmol·L^-1]
%  H2_th                : threshold H2 (not used directly in Monods; placeholder)
%  K_H2, K_SO4, K_CO2   : Monod half-saturation constants [mmol·L^-1]
%
% p(16): K_Form          : Formate Monod half-constant [mmol·L^-1]
%
% p(17..18): DECAY & CAP
%  b                    : biomass decay [day^-1]
%  Xmax                 : logistic cap for biomass [mmol·L^-1]
%
% p(19..27): TRANSFER / HENRY / SECHENOV
%  kla_H2, kla_CO2, kla_H2S [day^-1]
%  HcpScale_H2, HcpScale_CO2, HcpScale_H2S [-]: freshwater Henry scale factors
%  ks_H2, ks_CO2, ks_H2S   [-]: Sechenov (salting-out) coefficients (per ionic strength)
%
% p(28..30): Ca / FeS / HS_sat
%  k_diss_Ca            : Ca re-equilibration rate to Ca_eq [day^-1]
%  k_prec               : FeS precipitation rate (limited by Fe pool) [mmol·L^-1·day^-1]
%  HS_sat               : HS- supersaturation threshold for precipitation [mmol·L^-1]
%
% p(31..33): Peng–Robinson binary interaction coefficients k_ij
%  kH2CO2, kH2CH4, kCO2CH4 [-]: affect a_mix in EOS
%
% p(34..35): Initial carbonate system
%  CT0_fit, Alk0_fit    : fitted initial CT and Alk [mmol·L^-1]
%
% p(36): Day-9 H2 injection amount [mmol]
%
% p(37..38): PRODUCT INHIBITIONS
%  KI_Ac_prod           : acetate product inhibition constant [mmol·L^-1]
%  KI_Form_prod         : formate product inhibition constant [mmol·L^-1]
%
% p(39): k_CO2_feed     : (disabled; regulator not used when pulses active)
%
% p(40): Day-7 CO2 make-up amount [mmol] (CH4 forced to 99× CO2)
%
% p(41): k_s_org        : pre-H2 heterotrophic SR rate [day^-1]

% --- Nominal guesses and bounds (you can tune as needed) ---
p0_kin = [ 0.20, 0.25, 0.15, 0.08, 0.06, 0.05, 0.05, 0.03, ...
           0.2,  0.2,  0.2,  1e-8, 1e-3, 1e-3, 1e-3 ];
lb_kin = [ 1e-4, 1e-4, 1e-4, 1e-4, 0.01, 0.01, 0.01, 0.01, ...
           1e-3, 1e-3, 1e-3, 0.0,  1e-6, 1e-3, 1e-6 ];
ub_kin = [ 2.0,  2.0,  2.0,  2.0,  0.5,  0.5,  0.5,  0.5,  ...
           5.0,  5.0,  5.0,  1e-4, 0.1,  0.1,  0.1 ];

p0_Kform =    [1e-3];   lb_Kform = [1e-6];   ub_Kform = [1e-1];
p0_decay_cap = [0.01, 2.0]; lb_decay_cap = [0.001, 0.2]; ub_decay_cap = [0.10, 10.0];

% Transfer/Henry/Sechenov (with tighter kla_CO2 bounds)
p0_tr = [ 30.0, 0.15, 25.0, 1.00, 1.00, 1.00, 0.04, 0.12, 0.05 ];
lb_tr = [ 0.5,  0.02, 0.5,  0.95, 0.95, 0.95, 0.00, 0.00, 0.00 ];
ub_tr = [ 200.0,0.40,200.0, 1.05, 1.05, 1.05, 0.40, 0.40, 0.20 ];

p0_ca_prec_sat = [0.05, 0.02, 0.10]; lb_ca_prec_sat = [0.00, 0.00, 0.00]; ub_ca_prec_sat = [0.20, 1.50, 5.00];

p0_kij = [0.08, 0.02, 0.03]; lb_kij = [-0.05, -0.05, -0.02]; ub_kij = [0.20, 0.10, 0.10];

p0_CTAlk = [15.0, 6.0]; lb_CTAlk = [1.0, 2.0]; ub_CTAlk = [50.0, 10.0];

p0_H2inj = [ max(H2g_init_mmol, 90.0) ]; lb_H2inj = [10.0]; ub_H2inj = [150.0];

p0_inh = [50.0, 10.0]; lb_inh = [5.0, 1.0]; ub_inh = [200.0, 100.0];

p0_kco2feed = [0.0]; lb_kco2feed = [0.0]; ub_kco2feed = [0.0];

p0_day7CO2 = [0.0]; lb_day7CO2 = [0.0]; ub_day7CO2 = [120.0];

p0_k_s_org = [0.15]; lb_k_s_org = [0.01]; ub_k_s_org = [1.0];

% Combine all
p0 = [p0_kin, p0_Kform, p0_decay_cap, p0_tr, p0_ca_prec_sat, p0_kij, p0_CTAlk, ...
      p0_H2inj, p0_inh, p0_kco2feed, p0_day7CO2, p0_k_s_org];
lb = [lb_kin, lb_Kform, lb_decay_cap, lb_tr, lb_ca_prec_sat, lb_kij, lb_CTAlk, ...
      lb_H2inj, lb_inh, lb_kco2feed, lb_day7CO2, lb_k_s_org];
ub = [ub_kin, ub_Kform, ub_decay_cap, ub_tr, ub_ca_prec_sat, ub_kij, ub_CTAlk, ...
      ub_H2inj, ub_inh, ub_kco2feed, ub_day7CO2, ub_k_s_org];

%% 4) Fit options
fit_opts = optimoptions('lsqnonlin', ...
    'Display','iter', ...
    'MaxFunctionEvaluations', 32000, ...
    'FiniteDifferenceType','central');

% Pack problem
prob.env      = env;
prob.y0       = y0;
prob.t_exp    = t_exp;
prob.data_exp = data_exp;

% Solve least-squares
[p_fit,~,~,~,~,~,~] = lsqnonlin(@(p) residuals_mura(p, prob), p0, lb, ub, fit_opts);

% Save a small MAT with best parameters and IC for reuse
out.p_fit = p_fit; out.env = env; out.y0 = y0;
save('best_fit_params_mura_clean.mat','-struct','out');

%% 5) Final simulation & plotting
[t_sim, y_sim, seg] = simulate_mura(p_fit, prob);
env_use = envFromP(p_fit, env);
fprintf('Using volumes: Vg = %.3f L, Vl = %.3f L\n', env_use.Vg, env_use.Vl);

% Unpack states
nH2g  = y_sim(:,1); nCO2g = y_sim(:,2); nCH4g = y_sim(:,3); nH2Sg = y_sim(:,4);
H2aq  = y_sim(:,5); CT    = y_sim(:,6);
SO4   = y_sim(:,7); Form  = y_sim(:,8); Acet = y_sim(:,9); S_tot = y_sim(:,10);
X     = y_sim(:,11); Ca   = y_sim(:,12); FeS = y_sim(:,13); Fe_pool = y_sim(:,14); Alk = y_sim(:,15);

% Carbonate speciation and pH
CO2aq = zeros(size(t_sim)); HCO3 = CO2aq; CO3 = CO2aq; pH = CO2aq;
for k=1:length(t_sim)
    [pH(k), CO2aq(k), HCO3(k), CO3(k)] = carb_from_CT_Alk(max(CT(k),1e-12), max(Alk(k),0), max(S_tot(k),1e-12), env_use.T);
end

% Sulfide speciation vectors
[H2S_aq_vec, HS_aq_vec] = speciate_sulfide_vec(S_tot, pH, env_use.pKa_H2S);

% Dashboard
species = {'nH2_g','nCO2_g','nCH4_g','nH2S_g','H2(aq)','CO2(aq)','SO4','Formate','Acetate','S_{aq}^{tot}', ...
           'Biomass','Ca^{2+}','FeS','Fe_{pool}'};

figure('Name','Gases & Aqueous – Mura (36 °C)');
for i = 1:length(species)
    subplot(7,2,i); hold on; grid on; title(species{i});
    switch i
        case 1, plot(t_exp, prob.data_exp(:,1),'ro','DisplayName','H_2(g) Exp'); plot(t_sim,nH2g,'r-','DisplayName','Model'); ylabel('mmol (gas)');
        case 2, plot(t_exp, prob.data_exp(:,2),'go','DisplayName','CO_2(g) Exp'); plot(t_sim,nCO2g,'g-','DisplayName','Model'); ylabel('mmol (gas)');
        case 3, plot(t_sim,nCH4g,'b-','DisplayName','Model'); ylabel('mmol (gas)');
        case 4, plot(t_sim,nH2Sg,'k-','DisplayName','Model'); ylabel('mmol (gas)');
        case 5, plot(t_sim,H2aq,'b-','DisplayName','Model');   ylabel('mmol/L');
        case 6, plot(t_sim,CO2aq,'g-','DisplayName','Model');  ylabel('mmol/L');
        case 7, plot(t_exp, prob.data_exp(:,3),'bo','DisplayName','SO_4 Exp'); plot(t_sim,SO4,'b-','DisplayName','Model'); ylabel('mmol/L');
        case 8, plot(t_exp, prob.data_exp(:,4),'ko','DisplayName','Formate Exp'); plot(t_sim,Form,'k-','DisplayName','Model'); ylabel('mmol/L');
        case 9, plot(t_exp, prob.data_exp(:,5),'mo','DisplayName','Acetate Exp'); plot(t_sim,Acet,'m-','DisplayName','Model'); ylabel('mmol/L');
        case 10, plot(t_sim,S_tot,'c-','DisplayName','Model'); ylabel('mmol/L');
        case 11, plot(t_sim,X,'k-','DisplayName','Model');     ylabel('mmol/L');
        case 12, plot(t_exp, prob.data_exp(:,6),'co','DisplayName','Ca^{2+} Exp'); plot(t_sim,Ca,'c-','DisplayName','Model'); ylabel('mmol/L');
        case 13, plot(t_sim,FeS,'k-','DisplayName','Model');   ylabel('mmol/L');
        case 14, plot(t_sim,Fe_pool,'k-','DisplayName','Model'); ylabel('mmol/L');
    end
    xlabel('days'); legend('Location','best');
end

% pH(t)
figure('Name','pH evolution'); plot(t_sim, pH, 'k-', 'LineWidth',1.5);
grid on; xlabel('days'); ylabel('pH'); title('pH(t) from CT–Alk');

% Kinetic Rates reporter
rates = zeros(length(t_sim),5); % [r_meth, r_sulf_total, r_acet, r_form, r_prec]
for k=1:length(t_sim)
    rates(k,:) = rate_out(t_sim(k), y_sim(k,:).', p_fit, env_use);
end
figure('Name','Kinetic Rates');
plot(t_sim, rates(:,1),'r-', 'DisplayName','r_{meth}'); hold on; grid on;
plot(t_sim, rates(:,2),'b-', 'DisplayName','r_{sulf}');
plot(t_sim, rates(:,3),'g-', 'DisplayName','r_{acet}');
plot(t_sim, rates(:,4),'m-', 'DisplayName','r_{form}');
plot(t_sim, rates(:,5),'k-', 'DisplayName','r_{prec}');
xlabel('days'); ylabel('mmol/L/day'); title('Kinetic Rates'); legend;

% H2S headspace diagnostic (ϕ–γ eq vs model)
HcpH2S_eff_base = env_use.Hcp_H2S_fw * env_use.HcpScale_H2S;
P = env_use.Pset;
n_fixed = nH2g + nCO2g + nCH4g;
y_eq = zeros(size(t_sim));
for k=1:length(t_sim)
    I_k = ionic_strength_estimate(SO4(k), Form(k), Acet(k), HS_aq_vec(k));
    Hcp_H2S_salt = HcpH2S_eff_base * exp(env_use.ks_H2S * I_k);
    y_eq(k) = max(min((H2S_aq_vec(k) ./ Hcp_H2S_salt)./P, 0.999999), 1e-12);
end
nH2Sg_eq = (y_eq./(1 - y_eq)).* n_fixed;

figure('Name','H2S headspace diagnostic');
plot(t_sim, nH2Sg, 'b-', 'LineWidth',1.6, 'DisplayName','n_{H2S,g} (model)'); hold on;
plot(t_sim, nH2Sg_eq, 'r--', 'LineWidth',1.6, 'DisplayName','n_{H2S,g}^{eq} (ϕ–γ)');
xlabel('days'); ylabel('mmol'); title('Headspace H_2S vs equilibrium'); legend('Location','best'); grid on;

% Sulfur mass balance (piecewise Vl due to day-17 water addition)
S_gas_mmol = nH2Sg;
S_aq_mmol  = S_tot * env_use.Vl;
S_FeS_mmol = FeS   * env_use.Vl;
S_total_model = S_gas_mmol + S_aq_mmol + S_FeS_mmol;

t = t_sim; rs = rates(:,2); % total SR rate
Vl0 = env.Vl; Vl2 = env.Vl + env.dVl_day17;
i17 = find(t >= 17, 1, 'first');
S_prod_cum = zeros(size(t));
if isempty(i17)
    S_prod_cum = cumtrapz(t, rs) * Vl0;
else
    S_prod_cum(1:i17) = cumtrapz(t(1:i17), rs(1:i17)) * Vl0;
    inc2 = cumtrapz(t(i17:end), rs(i17:end)) * Vl2;
    S_prod_cum(i17:end) = S_prod_cum(i17) + inc2 - inc2(1);
end
S_total0 = S_total_model(1);
S_total_expected = S_total0 + S_prod_cum;

figure('Name','Sulfur Mass Balance');
subplot(2,1,1);
plot(t_sim, S_total_model, 'b-', 'LineWidth', 1.4, 'DisplayName','Model total S (mmol)'); hold on;
plot(t_sim, S_total_expected, 'r--', 'LineWidth', 1.4, 'DisplayName','Expected total S (mmol)');
xlabel('days'); ylabel('mmol'); title('Total sulfur conservation'); legend('Location','best'); grid on;
subplot(2,1,2);
plot(t_sim, S_total_model - S_total_expected, 'k-', 'LineWidth', 1.2);
xlabel('days'); ylabel('mmol'); title('Mass-balance error (Model - Expected)'); grid on;
fprintf('Sulfur mass-balance error @ t_end: %.6g mmol\n', S_total_model(end) - S_total_expected(end));

% Export .dat
fileID = fopen('mura_2024_inc_rates_clean.dat','w');
fprintf(fileID, ['Time(d) nH2_g nCO2_g nCH4_g nH2S_g H2_aq CO2_aq CT Alk HCO3 CO3 SO4 Formate Acetate S_tot ', ...
                 'H2S_aq HS_aq pH Biomass Ca FeS Fe_pool r_meth r_sulf r_acet r_form r_prec\n']);
fmt = ['%10.6f ', '%12.6g %12.6g %12.6g %12.6g ', ...
       '%12.6g %12.6g %12.6g %12.6g %12.6g %12.6g ', ...
       '%12.6g %12.6g %12.6g %12.6g ', ...
       '%12.6g %12.6g %10.6f ', ...
       '%12.6g %12.6g %12.6g %12.6g ', ...
       '%12.6g %12.6g %12.6g %12.6g %12.6g\n'];
for i = 1:length(t_sim)
    fprintf(fileID, fmt, ...
        t_sim(i), nH2g(i), nCO2g(i), nCH4g(i), nH2Sg(i), ...
        H2aq(i), CO2aq(i), CT(i), Alk(i), HCO3(i), CO3(i), ...
        SO4(i), Form(i), Acet(i), S_tot(i), ...
        H2S_aq_vec(i), HS_aq_vec(i), pH(i), ...
        X(i), Ca(i), FeS(i), Fe_pool(i), ...
        rates(i,1), rates(i,2), rates(i,3), rates(i,4), rates(i,5));
end
fclose(fileID);
fprintf('Wrote mura_2024_inc_rates_clean.dat\n');

out.t_sim = t_sim; out.y_sim = y_sim; out.rates = rates; out.seg = seg; out.pH = pH;
fprintf('\nDone. Staged timeline + CH4 cushion + day-7 99:1 mix + pre-H2 SR + smooth pulses + tight kla_CO2.\n');
end % main

% ---------------------------- Residuals ----------------------------
function res = residuals_mura(p, prob)
[t_sim, y_sim] = simulate_mura(p, prob);

% Interpolate onto experimental times
yH2g  = interp1(t_sim, y_sim(:,1), prob.t_exp, 'linear', 'extrap');
yCO2g = interp1(t_sim, y_sim(:,2), prob.t_exp, 'linear', 'extrap');
ySO4  = interp1(t_sim, y_sim(:,7), prob.t_exp, 'linear', 'extrap');
yForm = interp1(t_sim, y_sim(:,8), prob.t_exp, 'linear', 'extrap');
yAcet = interp1(t_sim, y_sim(:,9), prob.t_exp, 'linear', 'extrap');
yCa   = interp1(t_sim, y_sim(:,12), prob.t_exp, 'linear', 'extrap');

sim_mat = [yH2g, yCO2g, ySO4, yForm, yAcet, yCa];
exp_mat = prob.data_exp;

% Weighted log residuals; stronger weight on CO2(g)/SO4, early times
rngv = max(exp_mat,[],1) - min(exp_mat,[],1) + 1e-6;
wvar = [1.2, 2.0, 2.2, 1.8, 1.8, 1.5] ./ rngv;

log_sim = log1p(max(sim_mat,0));
log_exp = log1p(max(exp_mat,0));
residual = (log_sim - log_exp).*wvar;

% Time-window up-weighting for t <= 21 d
w_t = ones(size(prob.t_exp)); w_t(prob.t_exp <= 21) = 2.0;
residual = residual .* repmat(w_t(:),1,size(residual,2));

% Soft pH(0) penalty around ~6 ± 0.5 using fitted CT0/Alk0
CT0  = max(p(34),1e-6);
Alk0 = max(p(35),1e-6);
[pH0, ~, ~, ~] = carb_from_CT_Alk(CT0, Alk0, 1e-3, prob.env.T);
pen_pH0 = max(0, abs(pH0 - 6.0) - 0.5);

% Guard negative initial gases
y0neg = min(y_sim(1,[1 2 3 4]),0);
pen_neg = sum(abs(y0neg));

res = [residual(:); 10*pen_pH0; pen_neg];
end

% ----------------------- Simulator with events ----------------------
function [t_all, y_all, seg_info] = simulate_mura(p, prob)
env = envFromP(p, prob.env);

% Overwrite initial state with fitted CT0 & Alk0 and staged gases
y_curr = prob.y0;
y_curr(1) = 0.0;                              % H2=0 at t0 (H2 pulse supplies it later)
y_curr(2) = max(prob.env.CO2g_init_mmol,1e-6);% CO2(g) exact from data
y_curr(6) = max(p(34),1e-6);                  % CT0 fit
y_curr(15)= max(p(35),1e-6);                  % Alk0 fit

% Segment boundaries (include t_end of experiment)
t_ev = [0, env.t_events, prob.t_exp(end)];
t_all = []; y_all = []; seg_info = struct('tspan',{},'event','', 'delta',[]);

for k=1:length(t_ev)-1
    tspan = [t_ev(k), t_ev(k+1)];
    odes = @(t,y) odefun_mura(t,y,p,env);
    S = buildJPattern();
    opts = odeset('NonNegative',1:15,'RelTol',1e-6,'AbsTol',1e-9,'MaxStep',0.5,'JPattern',S);
    [t_seg, y_seg] = ode15s(odes, tspan, y_curr, opts);

    if isempty(t_all)
        t_all = t_seg; y_all = y_seg;
    else
        t_all = [t_all; t_seg(2:end)];
        y_all = [y_all; y_seg(2:end,:)];
    end

    if k < length(t_ev)-1
        y_curr = y_seg(end,:).';
        switch env.t_events(k)
            case 17
                % Day 17: water addition (discrete volume change)
                env.Vl = env.Vl + env.dVl_day17;
                seg_info(end+1).event = 'day17 water addition';
                seg_info(end).delta = env.dVl_day17;
                seg_info(end).tspan = sprintf('[%.2f, %.2f]', tspan(1), tspan(2));
            otherwise
                % day7/day9 feeds are handled inside ODE by smooth pulses
        end
    end
end
end

function S = buildJPattern()
S = sparse(15,15); S = S + speye(15);
S(1,[1 5])=1; S(2,[2 6])=1; S(3,3)=1; S(4,[4 10 13 15])=1;
S(5,[1 2 3 4 5 6 7 8 9 10 11 15])=1;
S(6,[2 5 8 9 15])=1;
S(7,[5 7 11 15])=1; S(8,[5 8 11])=1; S(9,[5 9 11])=1;
S(10,[5 7 10 13 4])=1; S(11,[5 8 9 11])=1;
S(12,12)=1; S(13,[10 13 14 15])=1; S(14,[13 14])=1; S(15,[7 13 15])=1;
end

% ------------------------------- ODE -------------------------------
function dydt = odefun_mura(t, y, p, env)
% Unpack states
nH2g = y(1); nCO2g = y(2); nCH4g = y(3); nH2Sg = y(4);
H2aq = y(5); CT    = y(6); SO4 = y(7); Form = y(8); Acet = y(9);
S_tot = y(10); X = y(11); Ca = y(12); FeS = y(13); Fe_pool = y(14); Alk = y(15);

env2 = envFromP(p, env);

% Gas composition & fugacity coefficients (PR EOS) for H2/CO2/CH4
ntot = max(nH2g+nCO2g+nCH4g+nH2Sg, 1e-12);
yH2 = nH2g/ntot; yCO2 = nCO2g/ntot; yCH4 = nCH4g/ntot; yH2S = nH2Sg/ntot;

phi = PR_phi_mix(env2.PR, [yH2,yCO2,yCH4], env2.Pset, env2.T); % ϕ for H2, CO2, CH4

% Sechenov ionic strength estimate for salting-out
I = ionic_strength_estimate(SO4, Form, Acet, sulfide_HS(S_tot, Alk, env2.pKa_H2S));
Hcp_H2_eff  = env2.Hcp_H2_fw  * env2.HcpScale_H2  * exp(env2.ks_H2  * I);
Hcp_CO2_eff = env2.Hcp_CO2_fw * env2.HcpScale_CO2 * exp(env2.ks_CO2 * I);
Hcp_H2S_eff = env2.Hcp_H2S_fw * env2.HcpScale_H2S * exp(env2.ks_H2S * I);

% Gas fugacities
fH2  = phi(1)*yH2 *env2.Pset;
fCO2 = phi(2)*yCO2*env2.Pset;
fH2S = yH2S*env2.Pset;  % PR for H2S omitted -> use y*P

% Liquid equilibrium concentrations
Ceq_H2  = Hcp_H2_eff  * fH2;
Ceq_CO2 = Hcp_CO2_eff * fCO2;
Ceq_H2S = Hcp_H2S_eff * fH2S;

% Mass transfer fluxes
J_H2 = env2.kla_H2 *(Ceq_H2  - max(H2aq,0));
[pH_now, CO2aq, ~, ~] = carb_from_CT_Alk(max(CT,1e-12), max(Alk,0), max(S_tot,1e-12), env.T);
J_CO2 = env2.kla_CO2*(Ceq_CO2 - CO2aq);
[H2S_aq, HS_aq] = speciate_sulfide(max(S_tot,1e-12), pH_now, env.pKa_H2S);
Jout_H2S = env2.kla_H2S*(H2S_aq - Ceq_H2S); % positive -> out to gas

% Regulator feed disabled (pulses used)
F_CO2_feed = 0.0;

% Monod terms & inhibitions (guard non-negatives)
H2aq  = max(H2aq,1e-12); CO2aq = max(CO2aq,1e-12); SO4 = max(SO4,1e-12);
Form  = max(Form,1e-12); Acet = max(Acet,0);       X   = max(X,1e-12);

f_inh_m = env2.KI_m/(env2.KI_m + HS_aq);
f_inh_s = env2.KI_s/(env2.KI_s + HS_aq);
f_inh_a = env2.KI_a/(env2.KI_a + HS_aq);

mH2   = H2aq /(env2.K_H2  + H2aq);
mSO4  = SO4  /(env2.K_SO4 + SO4 );
mCO2  = CO2aq/(env2.K_CO2 + CO2aq);
mFORM = Form /(env2.K_Form+ Form );

% Product inhibitions
f_inh_FormProd = 1/(1 + Form/ env2.KI_Form_prod);
f_inh_AcProd   = 1/(1 + Acet/ env2.KI_Ac_prod);

% Microbial succession switches (optional, per your original script)
t_meth_on = 35; k_act = 0.35;
f_meth = 1./(1 + exp(-k_act*(t - t_meth_on)));

SO4_th = 0.20; k_so4 = 20.0;
f_postSO4 = 1./(1 + exp(k_so4*(SO4 - SO4_th)));

% Main rates (mmol·L^-1·day^-1)
r_meth = f_meth .* env2.k_m .* X .* mH2  .* mCO2  .* f_inh_m;
r_sulf =          env2.k_s .* X .* mH2  .* mSO4  .* f_inh_s;
r_form = f_postSO4.*env2.k_f .* X .* mH2  .* mCO2  .* f_inh_a .* f_inh_FormProd;
r_acet = f_postSO4.*env2.k_a .* X .* mH2  .* mFORM .* f_inh_a .* f_inh_AcProd;

% FeS precipitation (limited by Fe_pool, triggered by HS supersaturation)
r_prec_raw = env2.k_prec * max(0, HS_aq - env2.HS_sat);
r_prec     = min(r_prec_raw, max(Fe_pool,0));

% Pre-H2 heterotrophic sulfate reduction (active before day 9)
f_preH2    = 1./(1 + exp(20*(t - 9)));
r_sulf_org = env2.k_s_org * X .* mSO4 .* f_inh_s .* f_preH2;

% Smooth pulses (mmol/day) — day7 CO2/CH4, day9 H2
smoothwin = @(tt, t0, tau, k) 1./(1+exp(-k*(tt - t0))) - 1./(1+exp(-k*(tt - (t0 + tau))));
w_CO2 = smoothwin(t, env2.p_CO2.t0, env2.tau_pulse, env2.k_pulse);
w_CH4 = smoothwin(t, env2.p_CH4.t0, env2.tau_pulse, env2.k_pulse);
w_H2  = smoothwin(t, env2.p_H2.t0,  env2.tau_pulse, env2.k_pulse);
F_CO2_pulse = (max(env2.p_CO2.dn,0)/max(env2.tau_pulse,eps)) * w_CO2;
F_CH4_pulse = (max(env2.p_CH4.dn,0)/max(env2.tau_pulse,eps)) * w_CH4;
F_H2_pulse  = (max(env2.p_H2.dn, 0)/max(env2.tau_pulse,eps)) * w_H2;

% Gas balances
dnH2g  = -(J_H2 )* env2.Vl + F_H2_pulse;
dnCO2g = -(J_CO2)* env2.Vl + F_CO2_pulse + F_CO2_feed;
dnCH4g = +(r_meth)* env2.Vl + F_CH4_pulse;
dnH2Sg = +(Jout_H2S)* env2.Vl;

% Liquid balances with chained stoichiometry (avoid double counting CO2 sinks)
dH2aq = J_H2 - 4*r_meth - 4*r_sulf - 1*r_form - 3*r_acet;
dCT   = J_CO2 - 1*r_meth - 1*r_form - 1*r_acet;
dForm = + r_form - r_acet;
dAcet = + r_acet;

% Other species
dSO4   = - r_sulf - r_sulf_org;
dStot  = + r_sulf + r_sulf_org - r_prec - Jout_H2S;

% Biomass (sum of yields; logistic cap; decay)
G      = env2.Y_m*r_meth + env2.Y_s*r_sulf + env2.Y_a*r_acet + env2.Y_f*r_form;
f_log  = max(0, 1 - X/env2.Xmax);
dX     = G * f_log - env2.b * X;

% Ca & Fe pools
dCa      = env2.k_diss_Ca * max(env2.Ca_eq - Ca, 0);
dFeS     = + r_prec;
dFe_pool = - r_prec;

% Alkalinity: SR (+1 eq) + SR_org (+1 eq) − FeS precipitation (−1 eq)
dAlk   = + r_sulf + r_sulf_org - r_prec;

dydt = [dnH2g; dnCO2g; dnCH4g; dnH2Sg; dH2aq; dCT; dSO4; dForm; dAcet; dStot; dX; dCa; dFeS; dFe_pool; dAlk];
end

% --------------------------- Rate reporter --------------------------
function dr = rate_out(t, y, p, env)
env2 = envFromP(p, env);
H2aq  = max(y(5),1e-12); CT = max(y(6),1e-12); SO4 = max(y(7),1e-12);
Form  = max(y(8),1e-12); Acet = max(y(9),0);   S_tot = max(y(10),1e-12);
X     = max(y(11),1e-12); Fe_pool = max(y(14),0); Alk = max(y(15),0);

[pH_now, CO2aq, ~, ~] = carb_from_CT_Alk(CT, Alk, S_tot, env.T);
[~, HS_aq] = speciate_sulfide(S_tot, pH_now, env.pKa_H2S);

f_inh_m = env2.KI_m/(env2.KI_m + HS_aq);
f_inh_s = env2.KI_s/(env2.KI_s + HS_aq);
f_inh_a = env2.KI_a/(env2.KI_a + HS_aq);

mH2   = H2aq /(env2.K_H2  + H2aq);
mSO4  = SO4  /(env2.K_SO4 + SO4 );
mCO2  = CO2aq/(env2.K_CO2 + CO2aq);
mFORM = Form /(env2.K_Form+ Form );

f_inh_FormProd = 1/(1 + Form/ env2.KI_Form_prod);
f_inh_AcProd   = 1/(1 + Acet/ env2.KI_Ac_prod);

t_meth_on = 35; k_act = 0.35;
f_meth = 1./(1 + exp(-k_act*(t - t_meth_on)));

SO4_th = 0.20; k_so4 = 20.0;
f_postSO4 = 1./(1 + exp(k_so4*(SO4 - SO4_th)));

f_preH2    = 1./(1 + exp(20*(t - 9)));

r_meth = f_meth .* env2.k_m .* X .* mH2  .* mCO2  .* f_inh_m;
r_sulf =          env2.k_s .* X .* mH2  .* mSO4  .* f_inh_s;
r_form = f_postSO4.*env2.k_f .* X .* mH2  .* mCO2  .* f_inh_a .* f_inh_FormProd;
r_acet = f_postSO4.*env2.k_a .* X .* mH2  .* mFORM .* f_inh_a .* f_inh_AcProd;
r_sulf_org = env2.k_s_org .* X .* mSO4 .* f_inh_s .* f_preH2;

r_prec_raw = env2.k_prec * max(0, HS_aq - env2.HS_sat);
r_prec     = min(r_prec_raw, Fe_pool);

dr = [r_meth, r_sulf + r_sulf_org, r_acet, r_form, r_prec];
end

% ----------------------- Map p -> env2 (metadata) --------------------
function env2 = envFromP(p, env)
env2 = env;

% (1..15) kinetics & Monods
env2.k_m  = p(1);  env2.k_s  = p(2);  env2.k_a = p(3);  env2.k_f = p(4);
env2.Y_m  = p(5);  env2.Y_s  = p(6);  env2.Y_a = p(7);  env2.Y_f = p(8);
env2.KI_m = p(9);  env2.KI_s = p(10); env2.KI_a = p(11);
env2.H2_th= p(12);
env2.K_H2 = p(13); env2.K_SO4 = p(14); env2.K_CO2 = p(15);
env2.K_Form = p(16);

env2.b     = p(17);
env2.Xmax  = p(18);

% (19..27) transfer/Henry/Sechenov
env2.kla_H2  = p(19); env2.kla_CO2 = p(20); env2.kla_H2S = p(21);
env2.HcpScale_H2  = p(22); env2.HcpScale_CO2 = p(23); env2.HcpScale_H2S = p(24);
env2.ks_H2   = p(25); env2.ks_CO2  = p(26); env2.ks_H2S  = p(27);

% (28..30) Ca/FeS/HS_sat
env2.k_diss_Ca = p(28); env2.k_prec = p(29); env2.HS_sat = p(30);

% (31..33) PR kij (H2-CO2, H2-CH4, CO2-CH4)
kH2CO2 = p(31); kH2CH4 = p(32); kCO2CH4 = p(33);
env2.PR.kij = [ 0,      kH2CO2, kH2CH4; ...
                kH2CO2, 0,      kCO2CH4; ...
                kH2CH4, kCO2CH4, 0 ];

% (34..35) fitted CT0 & Alk0 -> applied in simulate_mura

% (37..38) product inhibitions
env2.KI_Ac_prod   = p(37);
env2.KI_Form_prod = p(38);

% (39) CO2 regulator coefficient (unused when pulses active)
env2.k_CO2_feed = p(39);

% (40) Day-7 CO2 make-up (CH4 forced to 99× CO2)
env2.p_CO2_dn = max(p(40),0);

% (41) pre-H2 heterotrophic SR
env2.k_s_org = p(41);

% Pulse meta (used in ODE for smooth feeds)
env2.tau_pulse = env.tau_pulse;
env2.k_pulse   = env.k_pulse;

% Event pulse objects
env2.p_CO2 = struct('t0',7, 'dn', env2.p_CO2_dn);
env2.p_CH4 = struct('t0',7, 'dn', 99*env2.p_CO2_dn);
env2.p_H2  = struct('t0',9, 'dn', max(p(36),0));
end

% ----------------------- Peng–Robinson helpers ----------------------
function PR = getPRparams(T)
PR.comp  = {'H2','CO2','CH4'};
PR.Tc    = [ 33.19, 304.2, 190.6];
PR.Pc    = [ 12.98, 73.8,  45.99];
PR.omega = [-0.220, 0.225, 0.011];
PR.kij   = [0 0 0; 0 0 0; 0 0 0];
R = 0.08314472; % bar·L·(mol^-1·K^-1)
Tr = T ./ PR.Tc;
m = 0.37464 + 1.54226.*PR.omega - 0.26992.*PR.omega.^2;
alpha = (1 + m.*(1 - sqrt(Tr))).^2;
PR.a_i = 0.45724 * (R.^2) .* (PR.Tc.^2)./ PR.Pc .* alpha;
PR.b_i = 0.07780 * R .* PR.Tc ./ PR.Pc;
end

function phi = PR_phi_mix(PR, zi, P, T)
R = 0.08314472;  % bar·L·(mol^-1·K^-1)
Pbar = P;
a_i  = PR.a_i; b_i = PR.b_i; kij = PR.kij; nc = numel(zi);
a_mix = 0.0; a_mix_i = zeros(1,nc);
for i=1:nc
    s=0;
    for j=1:nc
        t = sqrt(a_i(i)*a_i(j))*(1 - kij(i,j));
        a_mix = a_mix + zi(i)*zi(j)*t;
        s = s + zi(j)*t;
    end
    a_mix_i(i) = s;
end
b_mix = sum(zi .* b_i);
A = a_mix*Pbar/(R^2*T^2);
B = b_mix*Pbar/(R*T);
Z = cubic_vapor_root_PR(A,B);
Bi = b_i * Pbar/(R*T);
phi = zeros(1,nc);
for i=1:nc
    lnphi = Bi(i)*(Z-1)/B ...
          - log(Z - B) ...
          + A/(2*sqrt(2)*B) * ( 2*a_mix_i(i)/a_mix - Bi(i) ) ...
            * log( (Z + (1+sqrt(2))*B) / (Z + (1-sqrt(2))*B) );
    phi(i) = exp(lnphi);
end
end

function Z = cubic_vapor_root_PR(A,B)
c2 = -(1 - B); c1 = A - 3*B^2 - 2*B; c0 = -(A*B - B^2 - B^3);
zl = max(B + 1e-12, 1e-8); zu = 10;
fl = (zl*(zl*(zl + c2) + c1) + c0);
fu = (zu*(zu*(zu + c2) + c1) + c0);
for k=1:10
    zm = 0.5*(zl+zu);
    fm = (zm*(zm*(zm + c2) + c1) + c0);
    if fl*fm <= 0, zu=zm; fu=fm; else, zl=zm; fl=fm; end
end
Z = zm;
for k=1:6
    f  = (Z*(Z*(Z + c2) + c1) + c0);
    df = (3*Z + 2*c2)*Z + c1;
    step = -f/max(df,1e-16);
    Z = max(B + 1e-12, Z + step);
end
end

% ----------------------- Carbonate system helpers -------------------
function [pH, CO2aq, HCO3, CO3] = carb_from_CT_Alk(CT, Alk, S_tot, T)
% Robust CT–Alk solver (bisection on charge balance)
pK1 = 6.30; pK2 = 10.30; pKw = 14.00;
K1 = 10^-pK1; K2 = 10^-pK2; Kw = 10^-pKw;
H_low = 1e-12; H_high = 1e-3;

F_low = charge_residual(H_low, CT, Alk, S_tot, K1, K2, Kw);
F_high= charge_residual(H_high,CT, Alk, S_tot, K1, K2, Kw);
if F_low*F_high > 0
    H_low = 1e-14; F_low  = charge_residual(H_low,  CT, Alk, S_tot, K1, K2, Kw);
    H_high= 1e-2;  F_high = charge_residual(H_high, CT, Alk, S_tot, K1, K2, Kw);
end
for it = 1:40
    H_mid = 0.5*(H_low + H_high);
    F_mid = charge_residual(H_mid, CT, Alk, S_tot, K1, K2, Kw);
    if F_low*F_mid <= 0
        H_high = H_mid; F_high = F_mid;
    else
        H_low  = H_mid; F_low  = F_mid;
    end
    if abs(H_high - H_low) < 1e-14, break; end
end
H = max(1e-14, min(1e-2, 0.5*(H_low + H_high)));
pH = -log10(H);
denom = H^2 + K1*H + K1*K2;
CO2aq = CT * (H^2 / denom);
HCO3  = CT * (K1*H / denom);
CO3   = CT * (K1*K2 / denom);
end

function F = charge_residual(H, CT, Alk, S_tot, K1, K2, Kw)
denom = H^2 + K1*H + K1*K2;
HCO3 = CT * (K1*H/denom);
CO3  = CT * (K1*K2/denom);
OH   = Kw / H;
pKaS = 6.95; KaS = 10^-pKaS;
HS   = S_tot * (KaS/(KaS+H));
Alk_calc = HCO3 + 2*CO3 + HS + OH - H;
F = Alk_calc - Alk;
end

function Alk0 = alk_from_CT_pH(CT, pH, S_tot, env)
H = 10^-pH;
pK1 = 6.30; pK2 = 10.30; pKw=14.00;
K1=10^-pK1; K2=10^-pK2; Kw=10^-pKw;
denom = H^2 + K1*H + K1*K2;
HCO3 = CT * (K1*H/denom);
CO3  = CT * (K1*K2/denom);
OH   = Kw / H;
KaS  = 10^-env.pKa_H2S;
HS   = S_tot * (KaS/(KaS+H));
Alk0 = HCO3 + 2*CO3 + HS + OH - H;
end

% -------------------------- Sulfide helpers -------------------------
function [H2S_aq, HS_aq] = speciate_sulfide(S_tot, pH, pKa)
Ka = 10^-pKa; H = 10^-pH;
HS_aq  = S_tot * (Ka/(Ka+H));
H2S_aq = S_tot - HS_aq;
end

function [H2S_aq_vec, HS_aq_vec] = speciate_sulfide_vec(S_tot_vec, pH_vec, pKa)
H2S_aq_vec = zeros(size(S_tot_vec));
HS_aq_vec  = H2S_aq_vec;
for k=1:numel(S_tot_vec)
    [H2S_aq_vec(k), HS_aq_vec(k)] = speciate_sulfide(S_tot_vec(k), pH_vec(k), pKa);
end
end

function HS = sulfide_HS(S_tot, Alk, pKa) %#ok<INUSD>
% Placeholder for ionic strength calculation; HS is handled elsewhere
HS = 0;
end

% ----------------------- Ionic strength estimate --------------------
function I = ionic_strength_estimate(SO4_mM, Form_mM, Acet_mM, HS_mM)
% I ≈ 1/2 Σ c_j z_j^2 (mol·L^-1), dominant ions: SO4^2-, HCOO-, CH3COO-, HS-
cSO4 = max(SO4_mM,0)/1000; zSO4 = -2;
cFor = max(Form_mM,0)/1000; zFor = -1;
cAce = max(Acet_mM,0)/1000; zAce = -1;
cHS  = max(HS_mM,0)/1000;   zHS  = -1;
I = 0.5*( cSO4*zSO4^2 + cFor*zFor^2 + cAce*zAce^2 + cHS*zHS^2 );
end


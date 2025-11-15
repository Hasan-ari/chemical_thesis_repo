function rnn_transport_multiguild_uq_v3
% - Mechanistic ODEs for microbial guilds (methanogenesis, sulfate reducers, acetogenesis)
% - Parameter fitting to lab data (H₂, CH₄, H₂S, SO₄, CO₂)
% - LSTM neural network to emulate microbial reaction dynamics in a 1D transport column
% - Reactive transport simulation with advection, dispersion, and uncertainty quantification

    %% Load experimental data
    raw = readmatrix('Muller_2024_H2_Sandstone_at_25C.txt');
    t_exp = raw(:,1); % time in days
    % concentrations of chemical species (H2, CO2, CH4, H2S) at columns from 2 to 5 are in µmol, at column 6 (SO4) is in already in mmol  
    data_exp = [raw(:,2:5)*1e-3, raw(:,7)]; % Convert µmol to mmol

    %% Initial condition: [H2, CO2, CH4, H2S, SO4, FeS (Precipitated iron sulfide)
    % , X_meth (Methanogen biomass), X_sulf (Sulfate reducer biomass), X_aceto (Acetogen biomass), Acetate (Acetic acid)]
    x0 = [data_exp(1,:)'; 0.01; 0.01; 0.01; 0; 0]; % 10 elements 

    %% Fit mechanistic parameters and simulate trajectory
    p_fit = fit_mechanistic_params(t_exp, data_exp, x0);
    t = linspace(0, t_exp(end), 2000);
    [~, xTrain] = ode45(@(t,y) trueODEfunc_multiguild(t,y,p_fit), t, x0);
    xTrain = xTrain'; % [10 x timeSteps]

    %% Prepare sequences for RNN training
    sequenceLength = 10;
    X = {}; Y = {};
    for i = 1:(size(xTrain,2) - sequenceLength)
        X{end+1} = xTrain(:,i:i+sequenceLength-1);
        Y{end+1} = xTrain(:,i+sequenceLength);
    end

    %% Define RNN with dropout
    % layers = [ ...
    %     sequenceInputLayer(10)
    %     dropoutLayer(0.2)
    %     % rnnLayer(64,'OutputMode','last')
    %     SimpleRNNLayer(64,'my_rnn')
    %     dropoutLayer(0.2)
    %     fullyConnectedLayer(10)
    %     regressionLayer];
    % 
    % options = trainingOptions('adam', ...
    %     'MaxEpochs', 300, ...
    %     'MiniBatchSize', 64, ...
    %     'InitialLearnRate', 1e-3, ...
    %     'Shuffle','every-epoch', ...
    %     'Verbose',false);
    % 
    % net = trainNetwork(X,Y,layers,options);
    % save('trained_RNN_multiguild.mat','net');
%% Prepare sequences for LSTM training (sequence-to-one)
sequenceLength = 10; % number of past steps to use as input
X = {}; % cell array of sequences
Y = []; % numeric matrix of next-step targets

for i = 1:(size(xTrain,2) - sequenceLength)
    % Each predictor is a [features x sequenceLength] matrix
    X{end+1} = xTrain(:, i:i+sequenceLength-1);

    % Each response is a single [1 x features] row vector
    Y(end+1, :) = xTrain(:, i+sequenceLength)'; 
end

% Define LSTM network
layers = [ ...
    sequenceInputLayer(size(xTrain,1))  % number of features = number of states
    dropoutLayer(0.2)
    lstmLayer(64,'OutputMode','last')   % output only last time step
    dropoutLayer(0.2)
    fullyConnectedLayer(size(xTrain,1)) % predict all features at next step
    regressionLayer];

% Training options
options = trainingOptions('adam', 'MaxEpochs', 300, 'MiniBatchSize', 64, 'InitialLearnRate', 1e-3, 'Shuffle','every-epoch', ...
    'Verbose',false);

% Train sequence-to-one model
net = trainNetwork(X, Y, layers, options);

% Save trained network
save('trained_LSTM_multiguild.mat','net');
   
% --------------------------1D REDOX REACTIVE TRANSPORT ZONATION------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Spatial and temporal discretization--------------------------------------
 %% Transport simulation with ensemble sampling
L=75;                                         % length of column [m]
N=75;                                         % Number of cells
cell_w=1;                                     % Cell width [m]
S_time=t_exp(end);                                   % Simulation time [d]
% Spatial Discretization --------------------------------------------------
 x=cell_w:cell_w:L;
 nx = length(x);
% Flow and transfer parameters---------------------------------------------
% Confined aquifer
n=0.3;                                       % porosity [-]
q=1;                                         % Darcys velocity [m/d]
v=q/n;                                       % seepage velocity [m/d]
D=0.3;                                       % Dispersion coefficient [m^2/d]
alpha=D/v;                                   % dispersivity [m]

% Travel time in each cell -------------------------------------------------
dt=cell_w/v;                                       % [d]

% Matrix of Concentrations 
% Rows related to length coordinates
% Columns related to components
    cmob = zeros(nx, 6); % H2, CO2, CH4, H2S, SO4, Acetate, % mobile species
    cimob = zeros(nx, 4); % FeS, X_meth, X_sulf, X_aceto  % imobile species
    cmob(:,1:2) = 1e-4; cmob(:,5) = 5e-5;
    cimob(:,1:4) = 0.01;

    % Breakthrough curve (BTC) matrix for 10 species
    BTC_mean = zeros(0,10); BTC_std = zeros(0,10);

    historyLength = sequenceLength;
    historyBuffer = repmat(x0,1,historyLength);
%% Loop over all timepoints------------------------------------------------
    for time = 0:1:S_time

    %% -------------------------ADVECTION----------------------------------------
% Advection a Courant-number 1 implies that the concentrations are
% moved by exactly one box. The values in the last box are moved out.
% The first box receives the inflow concentration.
% Shifting mobile concentrations to the beginning of matrix
        cmob(2:end,:) = cmob(1:end-1,:);
        cmob(1,:) = [1e-4, 1e-4, 0, 0, 5e-5, 0];
%% ------------------------DISPERSION----------------------------------------

% Calculation of dispersive fluxes at the interior interfaces-------------

        Jd = (cmob(1:end-1,:) - cmob(2:end,:)) / cell_w * D;
% Add a dispersive flux of zero at the inflow boundary and assume that
% the dispersive flux at the outflow is identical to that at the last
% internal interface
        Jd = [zeros(1,6); Jd; Jd(end,:)];
%  Concentration change due to divergence of dispersive flux--------------        
        cmob = cmob + dt/cell_w * (Jd(1:end-1,:) - Jd(2:end,:));
%% ---------------------------REACTION-------------------------------------
% compute rate of change due to reaction-----------------------------------
% A new concentration zero matrix for using after ODE
        cmat_ensemble = zeros(nx,10,20); % 20 samples
        % For the ODE solver it creates matrix for each time duration  
        for it = 1:nx
            currentState = [cmob(it,:), cimob(it,:)]';
            historyBuffer = [historyBuffer(:,2:end), currentState];

            for s = 1:20
                y_pred = predict(net, historyBuffer, 'ExecutionEnvironment','cpu');
                cmat_ensemble(it,:,s) = y_pred;
            end
        end

        cmat_mean = mean(cmat_ensemble,3);
        cmat_std = std(cmat_ensemble,0,3);

        cmob = cmat_mean(:,1:6);
        cimob = cmat_mean(:,7:10);

        BTC_mean = [BTC_mean; cmat_mean(25,:)]; % Collect data at 25 m
        BTC_std = [BTC_std; cmat_std(25,:)];  % Collect data at 25 m
    end

    %% Plot BTC with uncertainty
    tvec = 0:1:S_time;
    species = {'H2','CO2','CH4','H2S','SO4','Acetate','FeS','X_meth','X_sulf','X_aceto'};

    figure;
    for i = 1:10
        subplot(5,2,i)
        % shadedErrorBar(tvec, BTC_mean(:,i), BTC_std(:,i), 'lineProps', '-b');
        % title(['BTC: ', species{i}]); xlabel('Time [d]'); ylabel('mmol/L');
        % Compute upper and lower bounds
upper = BTC_mean(:,i) + BTC_std(:,i);
lower = BTC_mean(:,i) - BTC_std(:,i);

% Fill the shaded area
fill([tvec fliplr(tvec)], [upper' fliplr(lower')], [0.8 0.8 1], ...
     'EdgeColor','none', 'FaceAlpha',0.3); 
hold on;

% Plot the mean line
plot(tvec, BTC_mean(:,i), '-b', 'LineWidth', 1.5);
% if i<=5
% plot(t_exp(:,1) ,data_exp(:,i), 'ro', 'LineWidth', 1.5)
% else
% end
title(['BTC: ', species{i}]); xlabel('Time [d]'); ylabel('mmol/L');
    end
end


function p_fit = fit_mechanistic_params(t_exp, data_exp, x0)
    % Define parameter bounds and initial guess
    p0 = [1, 1, 1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, -10];
    lb = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0, 0, 0, -50];
    ub = [10, 10, 10, 0.5, 0.5, 0.5, 10, 10, 10, 1, 1, 1, 0];

    % Comparison-friendly pre-fit printout
    fprintf('\n%s\n','======================================================================');
    fprintf('%s\n','PARAMETER FITTING: Nonlinear Least Squares Optimization (MATLAB)');
    fprintf('%s\n','======================================================================');
    fprintf('Algorithm: lsqnonlin (Trust-Region-Reflective)\n');
    fprintf('Residual length (data points flattened): %d\n', numel(data_exp));
    fprintf('Weights: [1, 1, 0.5, 0.5, 1] (H2, CO2, CH4, H2S, SO4)\n');
    names = {'k_meth','k_sulf','k_aceto','Y_m','Y_s','Y_a', ...
             'KI_meth','KI_sulf','KI_aceto','k_precip','H2S_sat','H2_thresh','DG_thresh'};
    fprintf('\nInitial guess (p0):\n');
    for i = 1:numel(p0)
        fprintf('  %-12s = %12.6f\n', names{i}, p0(i));
    end
    fprintf('\nLower bounds (lb):\n');
    for i = 1:numel(lb)
        fprintf('  %-12s >= %12.6f\n', names{i}, lb(i));
    end
    fprintf('\nUpper bounds (ub):\n');
    for i = 1:numel(ub)
        fprintf('  %-12s <= %12.6f\n', names{i}, ub(i));
    end
    fprintf('%s\n','======================================================================');

    options = optimoptions('lsqnonlin','Display','iter','MaxFunctionEvaluations',5000);
    [p_fit, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(@(p) residuals_multiguild(p, t_exp, data_exp, x0), p0, lb, ub, options);

    % Comparison-friendly post-fit summary
    fprintf('\n%s\n','======================================================================');
    fprintf('%s\n','FITTING COMPLETE (MATLAB)');
    fprintf('%s\n','======================================================================');
    fprintf('Success (exitflag>0): %d\n', exitflag > 0);
    if isfield(output,'message'); fprintf('Message: %s\n', strtrim(output.message)); end
    if isfield(output,'funcCount'); fprintf('Function evaluations: %d\n', output.funcCount); end
    if isfield(output,'iterations'); fprintf('Iterations: %d\n', output.iterations); end
    if isfield(output,'algorithm'); fprintf('Algorithm: %s\n', output.algorithm); end
    fprintf('Resnorm (sum of squares):    %.6f\n', resnorm);
    fprintf('Cost (0.5 * resnorm):        %.6f\n', 0.5*resnorm);
    rms_resid = sqrt(resnorm / numel(residual));
    fprintf('RMS residual:                %.6f\n', rms_resid);
    fprintf('Residual length:             %d\n', numel(residual));

    fprintf('\nFitted Parameters:\n');
    fprintf('----------------------------------------------------------------------\n');
    for i = 1:numel(p_fit)
        fprintf('  %-12s = %12.6f\n', names{i}, p_fit(i));
    end
    fprintf('%s\n','======================================================================');
end


function res = residuals_multiguild(p, t_exp, data_exp, x0)
    try
        [t_sim, y_raw] = ode45(@(t,y) trueODEfunc_multiguild(t,y,p), t_exp, x0);
        y_sim = interp1(t_sim, y_raw, t_exp, 'linear');

        log_sim = log1p(y_sim(:,1:5)); % H2, CO2, CH4, H2S, SO4
        log_exp = log1p(data_exp);
        weights = [1, 1, 0.5, 0.5, 1];
        res = (log_sim - log_exp) .* weights;

        if any(y_sim(:) < -1e-6)
            res = res + 1e3 * abs(min(y_sim(:)));
        end
        res = res(:);
    catch ME
        warning('ODE solver failed in residuals_multiguild: %s', ME.message);
        res = 1e6 * ones(numel(data_exp), 1);
    end
end


%% Mechanistic ODE function for multi-guild system
function dydt = trueODEfunc_multiguild(~, y, p)
% State variables
    H2 = y(1); CO2 = y(2); CH4 = y(3); H2S = y(4);
    SO4 = y(5); FeS = y(6); X_meth = y(7); X_sulf = y(8); X_aceto = y(9); Acetate = y(10);
 % Parameters
 %  Maximum rates
    k_meth = p(1);  % methanogenesis reaction
    k_sulf = p(2);  % sulfate reduction reaction
    k_aceto = p(3);  % acetogenesis reaction
    % Biomass yields
    Y_m = p(4);   % biomas yields from methanogens
    Y_s = p(5);   % from sulfate reduction
    Y_a = p(6);   % from acetogens
    % Inhibition constants
    KI_meth = p(7);  
    KI_sulf = p(8);
    KI_aceto = p(9);
    % Precipitation rate and saturation
    k_precip = p(10);
    H2S_sat = p(11);
    H2_thresh = p(12);
    % ΔG threshold 
    DG_thresh = p(13);
 % Constants -- Dynamic ΔG via Nernst Equation
    R = 8.314e-3; T = 298.15; RT = R*T;
    DG0_meth = -130; DG0_sulf = -152; DG0_aceto = -95;
  % Inhibition
    f_inh_meth  = KI_meth  / (KI_meth  + H2S); % inhibition by H₂S
    f_inh_sulf  = KI_sulf  / (KI_sulf  + H2S);
    f_inh_aceto = KI_aceto / (KI_aceto + H2S);
     % Activation threshold
    f_activation = H2 / (H2 + H2_thresh); % low-H₂ suppression
% Avoid log of zero or negative   % Thermodynamic feasibility
    H2 = max(H2,1e-6); CO2 = max(CO2,1e-6); CH4 = max(CH4,1e-6);
    SO4 = max(SO4,1e-6); H2S = max(H2S,1e-6); Acetate = max(Acetate,1e-6);
% Reaction quotients
    Q_meth  = CH4     / (H2^4 * CO2);
    Q_sulf  = H2S     / (H2^4 * SO4);
    Q_aceto = Acetate / (H2^4 * CO2^2);
% Dynamic Gibbs energies
    DG_meth  = DG0_meth  + RT*log(Q_meth);
    DG_sulf  = DG0_sulf  + RT*log(Q_sulf);
    DG_aceto = DG0_aceto + RT*log(Q_aceto);
% Thermodynamic feasibility
    f_thermo_meth  = 1 / (1 + exp((DG_meth  - DG_thresh)/RT));
    f_thermo_sulf  = 1 / (1 + exp((DG_sulf  - DG_thresh)/RT));
    f_thermo_aceto = 1 / (1 + exp((DG_aceto - DG_thresh)/RT));
   % Reaction rates with thermodynamic scaling
    r_meth  = k_meth  * H2 * CO2^(-2) * f_inh_meth  * f_activation * f_thermo_meth;
    r_sulf  = k_sulf  * H2 * SO4      * f_inh_sulf  * f_activation * f_thermo_sulf;
    r_aceto = k_aceto * H2 * CO2^2    * f_inh_aceto * f_activation * f_thermo_aceto;
    r_precip = k_precip * max(0, H2S - H2S_sat);

    % Differential equations
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
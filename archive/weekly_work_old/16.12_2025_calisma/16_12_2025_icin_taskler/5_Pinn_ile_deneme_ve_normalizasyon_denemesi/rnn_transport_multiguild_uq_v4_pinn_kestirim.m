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

    % %% Fit mechanistic parameters and simulate trajectory
    % p_fit = fit_mechanistic_params(t_exp, data_exp, x0);
    % t = linspace(0, t_exp(end), 2000);
    % 
    % disp('p_fit : ')
    % disp(p_fit)
    % 
    % [~, xTrain] = ode45(@(t,y) trueODEfunc_multiguild(t,y,p_fit), t, x0);
    % xTrain = xTrain'; % [10 x timeSteps]

    % %% 3. PINN Verisi ile Eğitim Seti Oluşturma (Hybrid Approach)
    % % Klasik ODE fit yerine, Python'da eğittiğimiz PINN verisini yüklüyoruz.
    % % Bu sayede LSTM, "gerçekçi" (PINN tarafından düzeltilmiş) davranışları öğrenecek.
    % 
    % disp('PINN verisi yükleniyor...');
    % loaded_data = load('PINN_Perfect_Data.mat'); 
    % xTrain = loaded_data.xTrain; % PINN'den gelen [10 x 2000] matris
    % t = loaded_data.t;           % Zaman vektörü

    
    
    %% 3. PINN Verisi ile Eğitim Seti Oluşturma (Hybrid Approach)
    %Normalizasyon ekledik.Ve zincirleme tahminlemeyi de güncelledik.
    disp('PINN verisi yükleniyor...');
    loaded_data = load('PINN_Perfect_Data.mat'); 
    xTrain_raw = loaded_data.xTrain; % Ham veri
    t = loaded_data.t;
    
    % --- NORMALİZASYON EKLEMESİ ---
    % Veriyi normalize et (Z-Score Normalization: (x - mean) / std)
    mu = mean(xTrain_raw, 2);
    sig = std(xTrain_raw, 0, 2);
    
    % Sıfıra bölme hatasını engelle (sabit kalan maddeler için)
    sig(sig == 0) = 1; 
    
    xTrain = (xTrain_raw - mu) ./ sig; % Artık xTrain normalize edildi
    % ------------------------------


    % İsteğe bağlı: PINN verisini görelim
    figure; plot(t, xTrain(5,:)); title('PINN Tarafından Düzeltilmiş SO4');

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




%%
disp('Validasyon grafikleri hazırlanıyor...');

% 1. Hazırlık: Zaman Eksenini Ayarla
% LSTM ilk 10 veriyi (sequenceLength) okuyup 11.yi tahmin eder.
% Bu yüzden tahminlerimiz 10. adımdan sonrasını kapsar.
seqLen = 10;
t_val = t(seqLen+1:end);        % Tahmin yapılan zaman aralığı

disp('x_Train transpoze hali : ')
disp(xTrain)
num_predictions = length(t_val); % Kaç adım tahmin edeceğiz?

% Çizdirilecek Bileşen İsimleri (Dosyandaki sıraya göre)
bilesenler = {'H2', 'CO2', 'CH4 (Metan)', 'H2S (Sülfür)', 'SO4'};

%% 2. Kestirim A: Birbirinden Bağımsız Tahmin (One-Step Ahead)
% Mantık: LSTM'e her adımda GERÇEK geçmiş veriyi veriyoruz. "Kopya çekiyor".
% predict fonksiyonu varsayılan olarak bunu yapar.
Y_independent = predict(net, X, 'MiniBatchSize', 1);
Y_independent = Y_independent'; % Boyut düzeltme: [5 x Zaman] olsun

% %% 3. Kestirim B: Zincirleme Tahmin (Chained / Recursive)
% % Mantık: LSTM'e sadece İLK 10 adımı veriyoruz. Sonrasını kendi üretiyor.
% Y_chained = zeros(size(xTrain,1), num_predictions); % Sonuçları buraya yazacağız
% 
% % Başlangıç Yemi: İlk 10 gerçek veri
% current_input = xTrain(:, 1:seqLen); 
% 
% for k = 1:num_predictions
%     % a) Gelecek adımı tahmin et
%     next_step = predict(net, current_input, 'ExecutionEnvironment','cpu');
% 
%     % b) Tahmini kaydet
%     Y_chained(:, k) = next_step';
% 
%     % c) Pencereyi kaydır (ZİNCİRLEME KISMI BURASI)
%     % Eski veriyi at, kendi ürettiğin 'next_step'i sona ekle.
%     % Artık gerçek veriden koptuk, tamamen yapay zekanın hayaline kaldık.
%     current_input = [current_input(:, 2:end), next_step'];
% end

%% 3. Kestirim B: Zincirleme Tahmin (Chained / Recursive)
Y_chained_norm = zeros(size(xTrain,1), num_predictions); 

% Başlangıç Yemi: İlk 10 normalize veri
current_input = xTrain(:, 1:seqLen); 

for k = 1:num_predictions
    % a) Gelecek adımı tahmin et (Normalize uzayda)
    next_step = predict(net, current_input, 'ExecutionEnvironment','cpu');
    
    % b) Tahmini kaydet
    Y_chained_norm(:, k) = next_step';
    
    % c) Pencereyi kaydır
    current_input = [current_input(:, 2:end), next_step'];
end

% --- DENORMALİZASYON (Gerçek değerlere dön) ---
% Bağımsız tahmin için:
Y_independent = (Y_independent .* sig) + mu;

% Zincirleme tahmin için:
Y_chained = (Y_chained_norm .* sig) + mu;

% xTrain'i de grafik için eski haline getir (Sadece çizim için lazım)
xTrain_plot = (xTrain .* sig) + mu;



%% 4. Çizim (3 Çizgi + 1 Scatter)
figure('Name', 'Karşılaştırma', 'Color', 'w', 'Position', [100 100 1200 800]);

for i = 1:5 % İlk 5 bileşen (H2, CO2, CH4, H2S, SO4)
    subplot(2, 3, i); % 2 satır, 3 sütunluk yer aç
    hold on;
    
    % A) Experiment Data (Scatter) - Siyah Noktalar
    % Txt dosyasından gelen ham veri
    plot(t_exp, data_exp(:,i), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, 'DisplayName', 'Deneysel Veri (Gerçek)');
    
    % B) Simulated Output (ODE) - Mavi Çizgi
    % Matematiksel modelin ürettiği "Kusursuz Yol"
    plot(t, xTrain_plot(i,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Simülasyon ');
    
    % C) Independent Predict (LSTM One-Step) - Yeşil Kesikli
    plot(t_val, Y_independent(i,:), 'g--', 'LineWidth', 1.5, 'DisplayName', 'LSTM (Bağımsız Tahmin)');
    
    % D) Chained Predict (LSTM Recursive) - Kırmızı Noktalı
    plot(t_val, Y_chained(i,:), 'r:', 'LineWidth', 2, 'DisplayName', 'LSTM (Zincirleme Tahmin)');
    
    % Görsel Ayarlar
    title(bilesenler{i});
    xlabel('Zaman (Gün)');
    ylabel('Konsantrasyon (mM)');
    grid on;
    
    % Legend (Sadece son grafiğe koyalım, kalabalık olmasın)
    if i == 5
        legend('Location', 'best');
    end
end








   
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
    % [k_meth, k_sulf, k_aceto, Y_m, Y_s, Y_a, KI_m, KI_s, KI_a, k_precip, H2S_sat, H2_thr, DG_thr]
    p0 = [1, 1, 1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, -10];
    lb = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0, 0, 0, -50];
    ub = [10, 10, 10, 0.5, 0.5, 0.5, 10, 10, 10, 1, 1, 1, 0];

    options = optimoptions('lsqnonlin','Display','iter','MaxFunctionEvaluations',5000);
    p_fit = lsqnonlin(@(p) residuals_multiguild(p, t_exp, data_exp, x0), p0, lb, ub, options);
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
    catch
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





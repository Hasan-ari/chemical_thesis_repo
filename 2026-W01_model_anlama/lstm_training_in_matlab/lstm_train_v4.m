% ============================================================================
% LSTM EĞİTİMİ - V4 TWO-PHASE MODEL VERİLERİ İLE
% ============================================================================
% Bu kod:
% - anaerobic_model_two_phase_mixedSR_25C_v4.m kodunu çalıştırır
% - Fitted parameters'ı alır
% - ODE çözümünü oluşturur (eğitim verisi)
% - LSTM ağını eğitir
% - Sonuçları görselleştirir
% ============================================================================

function lstm_train_v4

    fprintf('=== LSTM Eğitimi Başlıyor (v4 Sandstone 25C) ===\n\n');

    %% 1. FİTTED PARAMETERS'I YÜKLE
    fprintf('[1/6] Fitted parameters yükleniyor...\n');

    % Ana kod klasörü
    code_folder = 'd:\chemical_thesis_repo\2026-W01_model_anlama\code\matlab\';

    % Fitted parameters dosyasını yükle
    param_file = fullfile(code_folder, 'best_fit_params_Basalt_25C.mat');

    if ~isfile(param_file)
        error(['Fitted parameters dosyası bulunamadı!\n', ...
               'Önce ana kodu çalıştır: anaerobic_model_two_phase_mixedSR_25C_v4.m']);
    end

    load(param_file, 'p_fit', 'env', 'y0');

    % model_mixed fonksiyonunu path'e ekle
    addpath(code_folder);

    fprintf('   ✓ p_fit yüklendi (28 parametre)\n');
    fprintf('   ✓ env yüklendi (Vg=%.3f, Vl=%.3f, T=%.2f K)\n', env.Vg, env.Vl, env.T);
    fprintf('   ✓ y0 yüklendi (14 state variable)\n');
    fprintf('   ✓ model_mixed fonksiyonu path''e eklendi\n\n');

    %% 2. ODE ÇÖZÜMÜ İLE EĞİTİM VERİSİ OLUŞTUR
    fprintf('[2/6] ODE çözümü ile eğitim verisi oluşturuluyor...\n');

    % Deneysel veri zamanlarını yükle
    data_file = fullfile(code_folder, 'Muller_2024_H2_Basalt_at_25C.txt');
    raw = readmatrix(data_file);
    t_exp = raw(:,1);

    % Dense grid oluştur (LSTM için daha fazla veri noktası)
    t_train = linspace(0, t_exp(end), 2000)'; % 2000 zaman noktası
    fprintf('   Zaman aralığı: 0 - %.1f gün\n', t_exp(end));
    fprintf('   Nokta sayısı: %d\n', length(t_train));

    % model_mixed fonksiyonu zaten ana kod çalıştığında workspace'de
    % ODE çözümü
    odes = @(t,y) model_mixed(t,y,p_fit,env);
    opts = odeset('NonNegative',1:14,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);

    fprintf('   ODE çözülüyor (ode15s)...\n');
    tic;
    [~, y_sim] = ode15s(odes, t_train, y0, opts);
    elapsed = toc;
    fprintf('   ✓ ODE çözüldü (%.2f saniye)\n', elapsed);
    fprintf('   Çözüm boyutu: [%d x %d]\n\n', size(y_sim,1), size(y_sim,2));

    % Transpose: [14 x 2000] (features x timeSteps)
    xTrain_raw = y_sim';

    %% 3. NORMALİZASYON VE LOG TRANSFORM
    fprintf('[3/8] Veri normalizasyonu yapılıyor...\n');

    % Küçük değerler için log1p transform (Acetate, bazı gaz fazları)
    % İndeksler: nH2S_g(4), Acetate(10), Lag(13)
    log_indices = [4, 10, 13];
    xTrain = xTrain_raw;
    xTrain(log_indices, :) = log1p(xTrain_raw(log_indices, :));

    fprintf('   Log1p transform uygulandı: nH2S_g, Acetate, Lag\n');

    % Z-score normalizasyon: (x - mean) / std
    xTrain_mean = mean(xTrain, 2); % [14 x 1]
    xTrain_std = std(xTrain, 0, 2); % [14 x 1]
    xTrain_std(xTrain_std < 1e-8) = 1; % Sıfır std'den kaçın

    xTrain_norm = (xTrain - xTrain_mean) ./ xTrain_std;

    fprintf('   Z-score normalizasyon uygulandı\n');
    fprintf('   Ortalama aralığı: [%.2e, %.2e]\n', min(xTrain_mean), max(xTrain_mean));
    fprintf('   Std aralığı: [%.2e, %.2e]\n\n', min(xTrain_std), max(xTrain_std));

    % Normalizasyon parametrelerini kaydet (denormalizasyon için gerekli)
    norm_params.mean = xTrain_mean;
    norm_params.std = xTrain_std;
    norm_params.log_indices = log_indices;

    %% 4. LSTM İÇİN SEQUENCE VERİSİ HAZIRLA
    fprintf('[4/8] LSTM için sequence verisi hazırlanıyor...\n');

    sequenceLength = 10; % Geçmişten 10 zaman adımı kullan
    X = {}; % Cell array: Her eleman [14 x 10] matris
    Y = []; % Numeric matrix: Her satır [1 x 14] (hedef state)

    for i = 1:(size(xTrain_norm,2) - sequenceLength)
        % Geçmiş 10 zaman adımı: [14 x 10] (NORMALİZE EDİLMİŞ)
        X{end+1} = xTrain_norm(:, i:i+sequenceLength-1);

        % Sonraki zaman adımı: [1 x 14] (NORMALİZE EDİLMİŞ)
        Y(end+1, :) = xTrain_norm(:, i+sequenceLength)';
    end

    fprintf('   Sequence sayısı: %d\n', length(X));
    fprintf('   Her sequence boyutu: [14 x %d]\n', sequenceLength);
    fprintf('   Hedef (Y) boyutu: [%d x 14]\n\n', size(Y,1));

    %% 5. LSTM AĞI TASARLA (GÜÇLENDİRİLMİŞ)
    fprintf('[5/8] LSTM ağı tasarlanıyor (güçlendirilmiş versiyon)...\n');

    numFeatures = size(xTrain_norm,1); % 14 state variable
    numHiddenUnits = 128; % 64 → 128 (daha fazla kapasite)

    layers = [ ...
        sequenceInputLayer(numFeatures, 'Name', 'input')
        dropoutLayer(0.1, 'Name', 'dropout1')  % 0.2 → 0.1 (daha az agresif)
        lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm')
        dropoutLayer(0.1, 'Name', 'dropout2')  % 0.2 → 0.1
        fullyConnectedLayer(numFeatures, 'Name', 'fc')
        regressionLayer('Name', 'output')
    ];

    fprintf('   Katmanlar (GÜÇLENDİRİLMİŞ):\n');
    fprintf('     1. sequenceInputLayer(%d)  -> 14 state variable\n', numFeatures);
    fprintf('     2. dropoutLayer(0.1)        -> %%10 regularization (azaltıldı)\n');
    fprintf('     3. lstmLayer(%d, last)      -> LSTM (64→128 arttırıldı)\n', numHiddenUnits);
    fprintf('     4. dropoutLayer(0.1)        -> %%10 regularization (azaltıldı)\n');
    fprintf('     5. fullyConnectedLayer(%d)  -> 14 state tahmin\n', numFeatures);
    fprintf('     6. regressionLayer()        -> MSE loss\n\n');

    %% 6. EĞİTİM AYARLARI (OPTİMİZE EDİLMİŞ)
    fprintf('[6/8] Eğitim başlıyor (optimize edilmiş hiperparametreler)...\n');

    options = trainingOptions('adam', ...
        'MaxEpochs', 500, ...              % 300 → 500 (daha uzun eğitim)
        'MiniBatchSize', 32, ...           % 64 → 32 (daha hassas gradyan)
        'InitialLearnRate', 5e-4, ...      % 1e-3 → 5e-4 (daha yavaş başlangıç)
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 150, ...    % 100 → 150 (daha geç düşüş)
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'VerboseFrequency', 30, ...
        'Plots', 'training-progress');

    fprintf('   Optimizer: Adam\n');
    fprintf('   Max epochs: 500 (300→500 arttırıldı)\n');
    fprintf('   Batch size: 32 (64→32 küçültüldü)\n');
    fprintf('   Learning rate: 5e-4 (1e-3→5e-4 azaltıldı)\n');
    fprintf('   LR drop: Her 150 epoch''ta 0.5x (100→150 uzatıldı)\n');
    fprintf('   Training başladı...\n\n');

    % LSTM'i eğit
    net = trainNetwork(X, Y, layers, options);

    fprintf('\n   ✓ Eğitim tamamlandı!\n\n');

    %% 7. SONUÇLARI KAYDET
    fprintf('[7/8] Sonuçlar kaydediliyor...\n');

    % Kayıt klasörü (lstm_training_in_matlab klasörüne geri dön)
    save_folder = 'd:\chemical_thesis_repo\2026-W01_model_anlama\lstm_training_in_matlab\';
    cd(save_folder);

    % Ağı kaydet (normalizasyon parametreleri de dahil!)
    net_file = fullfile(save_folder, 'trained_LSTM_v4_normalized.mat');
    save(net_file, 'net', 'sequenceLength', 'numFeatures', 'numHiddenUnits', 'norm_params');
    fprintf('   ✓ LSTM ağı kaydedildi (norm_params dahil): %s\n', net_file);

    % Eğitim verisini kaydet (opsiyonel, sonraki kullanım için)
    data_file_out = fullfile(save_folder, 'lstm_training_data_v4_normalized.mat');
    save(data_file_out, 't_train', 'xTrain_raw', 'xTrain_norm', 'X', 'Y', 'p_fit', 'env', 'y0', 'norm_params');
    fprintf('   ✓ Eğitim verisi kaydedildi: %s\n', data_file_out);

    %% 8. VALİDASYON (ODE vs LSTM - DENORMALİZE EDİLMİŞ)
    fprintf('\n[8/8] Validasyon: ODE vs LSTM (denormalizasyon yapılıyor)...\n');

    % İlk 100 sequence için tahmin yap (normalize edilmiş)
    n_test = min(100, length(X));
    Y_pred_norm = zeros(n_test, numFeatures);

    for i = 1:n_test
        Y_pred_norm(i,:) = predict(net, X{i});
    end

    % DENORMALİZASYON: LSTM tahminlerini orijinal ölçeğe geri getir
    Y_pred_denorm = Y_pred_norm .* norm_params.std' + norm_params.mean';

    % Log transform uygulanan değişkenleri geri çevir (expm1)
    Y_pred_denorm(:, norm_params.log_indices) = expm1(Y_pred_denorm(:, norm_params.log_indices));

    % ODE gerçeğini de denormalize et (karşılaştırma için)
    Y_true_norm = Y(1:n_test, :);
    Y_true_denorm = Y_true_norm .* norm_params.std' + norm_params.mean';
    Y_true_denorm(:, norm_params.log_indices) = expm1(Y_true_denorm(:, norm_params.log_indices));

    % RMSE hesapla (DENORMALİZE EDİLMİŞ veriler üzerinden!)
    rmse_denorm = sqrt(mean((Y_true_denorm - Y_pred_denorm).^2, 1));

    fprintf('RMSE (her state için - orijinal ölçekte):\n');
    state_names = {'nH2_g', 'nCO2_g', 'nCH4_g', 'nH2S_g', 'H2_aq', 'CO2_aq', ...
                   'SO4', 'FeS', 'X', 'Acetate', 'HCO3', 'S_tot', 'Lag', 'Fe_pool'};
    for i = 1:numFeatures
        fprintf('  %10s: %.6e\n', state_names{i}, rmse_denorm(i));
    end

    % Görselleştirme (DENORMALİZE EDİLMİŞ değerlerle!)
    figure('Name', 'LSTM Validasyon - ODE vs LSTM (Normalized)', 'Position', [100 100 1400 900]);
    for i = 1:14
        subplot(4,4,i)
        plot(1:n_test, Y_true_denorm(:,i), 'b-', 'LineWidth', 1.5, 'DisplayName', 'ODE (True)');
        hold on;
        plot(1:n_test, Y_pred_denorm(:,i), 'r--', 'LineWidth', 1.5, 'DisplayName', 'LSTM (Pred)');
        title(state_names{i});
        xlabel('Sequence #');
        ylabel('Value (mmol or mmol/L)');
        legend('Location', 'best');
        grid on;
    end
    sgtitle('LSTM Validation: First 100 Sequences (Denormalized - Original Scale)');

    % Figürü kaydet
    fig_file = fullfile(save_folder, 'lstm_validation_v4_normalized.png');
    saveas(gcf, fig_file);
    fprintf('\n✓ Validasyon grafiği kaydedildi: %s\n', fig_file);

    fprintf('\n=== LSTM Eğitimi Tamamlandı! ===\n');
    fprintf('Dosyalar:\n');
    fprintf('  - %s\n', net_file);
    fprintf('  - %s\n', data_file_out);
    fprintf('  - %s\n', fig_file);

end

%% -------------------- NESTED FUNCTION: model_mixed (from v4 code) --------------------
function dydt = model_mixed(t, y, p, env)
% ODE function - kopyalandı: anaerobic_model_two_phase_mixedSR_25C_v4.m

Vg=env.Vg; Vl=env.Vl; T=env.T; Rgas=env.Rgas;
Hcp_H2=env.Hcp_H2_eff; Hcp_CO2=env.Hcp_CO2_eff; Hcp_H2S=env.Hcp_H2S_eff;
pH=env.pH_fun(t); pKa=env.pKa_H2S;

% State
nH2_g=y(1); nCO2_g=y(2); nCH4_g=y(3); nH2S_g=y(4);
H2_aq=y(5); CO2_aq=y(6); SO4=y(7); FeS=y(8); X=y(9);
Ac=y(10); HCO3=y(11); S_tot=y(12); Lag=y(13);
Fe_pool = y(14);

% Parameters
k_m  = p(1);  k_s  = p(2);  k_a  = p(3);
Y_m  = p(4);  Y_s  = p(5);  Y_a  = p(6);
KI_m = p(7);  KI_s = p(8);  KI_a = p(9);
k_prec = p(10); HS_sat = p(11); H2_th = p(12); DG_th = p(13);
K_H2 = p(14); K_SO4 = p(15); K_CO2 = p(16);
kla_H2 = p(17); kla_CO2 = p(18); kla_H2S = p(19);
b = p(20); t_lag = p(21); w_lag = p(22);
k_diss_gyp = p(23); beta_SO4_m = p(24);

% Thermodynamics
RkJ=8.314e-3; RT=RkJ*T;
DG0_m=-130; DG0_s=-152; DG0_a=-95;

% Guards
eps=1e-12;
nH2_g=max(nH2_g,eps); nCO2_g=max(nCO2_g,eps); nCH4_g=max(nCH4_g,eps); nH2S_g=max(nH2S_g,eps);
H2_aq=max(H2_aq,eps); CO2_aq=max(CO2_aq,eps); SO4=max(SO4,eps); S_tot=max(S_tot,eps);
Ac=max(Ac,eps); HCO3=max(HCO3,eps); X=max(X,eps); Fe_pool=max(Fe_pool,0);

% Partial pressures
pH2  = (nH2_g /1000)  * Rgas * T / Vg;
pCO2 = (nCO2_g/1000)  * Rgas * T / Vg;
pH2S = (nH2S_g/1000)  * Rgas * T / Vg;

% Henry equilibria
Ceq_H2  = Hcp_H2  * pH2;
Ceq_CO2 = Hcp_CO2 * pCO2;
Ceq_H2S = Hcp_H2S * pH2S;

% Gas-liquid transfers
J_H2  = kla_H2  * (Ceq_H2  - H2_aq);
J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq);

% Sulfide speciation
frac_HS  = 1/(1+10^(pKa - pH));
frac_H2S = 1 - frac_HS;
HS_aq  = S_tot*frac_HS;
H2S_aq = S_tot*frac_H2S;

% H2S outgassing
Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S);

% Inhibitions & Activation
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

% Thermo gates
Q_s = 1;
Q_a = Ac / (H2_aq^4 * CO2_aq^2);
DG_s = DG0_s + RT*log(Q_s);
DG_m = DG0_m;
DG_a = DG0_a + RT*log(Q_a);

fT_s = 1/(1+exp((DG_s - DG_th)/RT));
fT_m = 1/(1+exp((DG_m - DG_th)/RT));
fT_a = 1/(1+exp((DG_a - DG_th)/RT));

% Sulfate vs methanogen competition
f_comp_m = 1 / (1 + beta_SO4_m * SO4);

% Biomass-mediated rates
r_meth  = k_m * X * mH2 * mCO2       * f_inh_m * f_act * fT_m * f_comp_m;
r_sulf  = k_s * X * mH2 * mSO4       * f_inh_s * f_act * fT_s;
r_aceto = k_a * X * mH2 * (mCO2.^2)  * f_inh_a * f_act * fT_a;

% Precipitation
r_prec_raw = k_prec * max(0, HS_aq - HS_sat);
r_prec     = min(r_prec_raw, Fe_pool);

% Gypsum dissolution
SO4_sat    = env.SO4_sat_gyp;
r_diss_gyp = k_diss_gyp * max(0, SO4_sat - SO4);

% Gas balances
dnH2_g  = - J_H2  * Vl;
dnCO2_g = - J_CO2 * Vl;
dnCH4_g = + r_meth * Vl;
dnH2S_g = + Jout_H2S * Vl;

% Liquid balances
dH2_aq  = + J_H2  - 4*r_meth - 4*r_sulf - 4*r_aceto;
dCO2_aq = + J_CO2 - 1*r_meth - 2*r_aceto;
dSO4    = - 1*r_sulf + r_diss_gyp;
dFeS    = + r_prec;
dX      = + Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X;
dAc     = + r_aceto;
dHCO3   = 0;

% Fe pool balance
dFe_pool = - r_prec;

% Total dissolved sulfide balance
dS_tot = + 1.00*r_sulf - r_prec - Jout_H2S;

% Lag tracker
dLag = (f_lag - Lag)/max(w_lag,1e-3);

% Collect derivatives
dydt = [dnH2_g; dnCO2_g; dnCH4_g; dnH2S_g; dH2_aq; dCO2_aq; dSO4; dFeS; dX; dAc; dHCO3; dS_tot; dLag; dFe_pool];
end

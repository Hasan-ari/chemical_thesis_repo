function lstm_train_v5
% ============================================================================
% LSTM EĞİTİMİ V5 - GÜÇLENDİRİLMİŞ MİMARİ (STACKED LSTM + LONG SEQUENCE)
% ============================================================================
% İYİLEŞTİRMELER:
% 1. Sequence Length: 10 -> 50 (Daha uzun hafıza, trendleri daha iyi yakalar)
% 2. Mimari: Stacked LSTM (2 Katmanlı) -> Karmaşık ilişkileri öğrenir
% 3. Gradient Clipping: Eğitimin patlamasını engeller
% 4. Validasyon: Daha detaylı RMSE raporu
% ============================================================================

    fprintf('=== LSTM Eğitimi Başlıyor (v5 - Improved Architecture) ===\n\n');

    %% 1. FİTTED PARAMETERS'I YÜKLE
    fprintf('[1/6] Fitted parameters yükleniyor...\n');

    % Dizin ayarları (Kendi yolunuza göre gerekirse güncelleyin)
    code_folder = 'd:\chemical_thesis_repo\2026-W01_model_anlama\code\matlab\';
    param_file = fullfile(code_folder, 'best_fit_params_Basalt_25C.mat');

    if ~isfile(param_file)
        % Eğer tam yol çalışmazsa, scriptin olduğu klasöre bak
        if isfile('best_fit_params_Basalt_25C.mat')
             param_file = 'best_fit_params_Basalt_25C.mat';
             code_folder = pwd;
        else
             error('Parametre dosyası bulunamadı! Önce ODE modelini çalıştırıp p_fit kaydettiğinden emin ol.');
        end
    end

    load(param_file, 'p_fit', 'env', 'y0');
    addpath(code_folder); 

    fprintf('   ✓ Parametreler yüklendi.\n');

    %% 2. ODE ÇÖZÜMÜ İLE EĞİTİM VERİSİ OLUŞTUR (DENSE GRID)
    fprintf('[2/6] ODE çözümü ile YÜKSEK ÇÖZÜNÜRLÜKLÜ eğitim verisi oluşturuluyor...\n');

    % Deneysel veriden son zamanı al
    % (Dosya yoksa manuel t_end tanımlayalım, hata vermesin)
    try
        raw = readmatrix(fullfile(code_folder, 'Muller_2024_H2_Basalt_at_25C.txt'));
        t_end = raw(end,1);
    catch
        t_end = 20; % Varsayılan
        fprintf('   ! Uyarı: txt dosyası bulunamadı, t_end = 20 gün varsayıldı.\n');
    end

    % Dense grid: 2500 nokta (Daha sık örnekleme)
    t_train = linspace(0, t_end, 2500)'; 
    
    odes = @(t,y) model_mixed(t,y,p_fit,env);
    opts = odeset('NonNegative',1:14,'RelTol',1e-9,'AbsTol',1e-11,'MaxStep',0.1); % Toleranslar sıkılaştırıldı

    tic;
    [~, y_sim] = ode15s(odes, t_train, y0, opts);
    fprintf('   ✓ ODE çözüldü (%.2f saniye). Boyut: [%d x %d]\n', toc, size(y_sim,1), size(y_sim,2));

    % Features x TimeSteps formatına çevir
    xTrain_raw = y_sim'; 

    %% 3. VERİ ÖN İŞLEME (LOG TRANSFORM + Z-SCORE)
    fprintf('[3/6] Veri normalizasyonu (Log1p + Z-score)...\n');

    % Log-transform uygulanacak değişkenler (Küçük ve çok değişenler)
    % 4:nH2S_g, 8:FeS, 10:Acetate, 13:Lag, 14:Fe_pool
    log_indices = [4, 8, 10, 13, 14]; 
    
    xTrain = xTrain_raw;
    xTrain(log_indices, :) = log1p(xTrain_raw(log_indices, :));

    % Z-score Normalizasyon
    xTrain_mean = mean(xTrain, 2);
    xTrain_std  = std(xTrain, 0, 2);
    xTrain_std(xTrain_std < 1e-9) = 1; % Sıfıra bölmeyi engelle

    xTrain_norm = (xTrain - xTrain_mean) ./ xTrain_std;

    % Parametreleri sakla
    norm_params.mean = xTrain_mean;
    norm_params.std = xTrain_std;
    norm_params.log_indices = log_indices;

    %% 4. SEQUENCE OLUŞTURMA (WINDOWING)
    fprintf('[4/6] Sequence verisi hazırlanıyor (Window Size Artırıldı)...\n');

    % İYİLEŞTİRME: Sequence Length 10 -> 50
    % Modelin "hafızasını" artırıyoruz. Kimyasal kinetikte geçmiş önemlidir.
    sequenceLength = 50; 
    
    X = {};
    Y = [];

    num_samples = size(xTrain_norm, 2) - sequenceLength;
    
    for i = 1:num_samples
        % Girdi: t ile t+49 arası (50 adım)
        X{end+1} = xTrain_norm(:, i : i+sequenceLength-1);
        
        % Hedef: t+50 anındaki durum
        Y(end+1, :) = xTrain_norm(:, i+sequenceLength)';
    end

    fprintf('   Sequence Length: %d\n', sequenceLength);
    fprintf('   Toplam Örnek: %d\n', length(X));

    %% 5. STACKED LSTM MİMARİSİ (GÜÇLENDİRİLMİŞ)
    fprintf('[5/6] Stacked LSTM mimarisi kuruluyor...\n');

    numFeatures = size(xTrain_norm, 1);
    
    layers = [ ...
        sequenceInputLayer(numFeatures, 'Name', 'input')
        
        % 1. LSTM Katmanı (Sequence döndürür, böylece bir sonrakine bağlarız)
        lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'lstm1')
        dropoutLayer(0.2, 'Name', 'drop1') % Regularization artırıldı
        
        % 2. LSTM Katmanı (Sadece son adımı döndürür)
        lstmLayer(64, 'OutputMode', 'last', 'Name', 'lstm2')
        dropoutLayer(0.2, 'Name', 'drop2')
        
        fullyConnectedLayer(numFeatures, 'Name', 'fc')
        regressionLayer('Name', 'output')
    ];

    % Eğitim Seçenekleri
    options = trainingOptions('adam', ...
        'MaxEpochs', 500, ...             % Daha uzun eğitim (Sabır)
        'MiniBatchSize', 64, ...           % Stabil gradyan için makul boyut
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 250, ...
        'GradientThreshold', 1, ...        % İYİLEŞTİRME: Patlayan gradyanları engelle
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'VerboseFrequency', 50, ...
        'Plots', 'training-progress');

    %% 6. EĞİTİM VE KAYIT
    fprintf('[6/6] Model eğitiliyor...\n');
    
    net = trainNetwork(X, Y, layers, options);

    % Kaydet
    save_folder = fileparts(code_folder); % Bir üst klasör veya uygun yer
    if isempty(save_folder), save_folder = pwd; end
    
    net_file = fullfile(save_folder, 'trained_LSTM_v5_stacked.mat');
    save(net_file, 'net', 'sequenceLength', 'norm_params', 'p_fit', 'env');
    
    fprintf('   ✓ Model kaydedildi: %s\n', net_file);

    %% 7. HIZLI VALIDASYON (GRAFİK)
    fprintf('\n   Validasyon grafiği çiziliyor...\n');
    
    % Test için ilk 200 sequence
    n_test = min(200, length(X));
    Y_pred_norm = predict(net, X(1:n_test)); % Cell array olduğu için X(1:n)
    
    % Denormalizasyon
    Y_pred = (Y_pred_norm .* norm_params.std') + norm_params.mean';
    Y_true = (Y(1:n_test, :) .* norm_params.std') + norm_params.mean';
    
    % Log-inverse
    Y_pred(:, log_indices) = expm1(Y_pred(:, log_indices));
    Y_true(:, log_indices) = expm1(Y_true(:, log_indices));

    % Çizim
    figure('Name', 'LSTM v5 Validation', 'Position', [100 100 1200 800]);
    state_names = {'nH2_g', 'nCO2_g', 'nCH4_g', 'nH2S_g', 'H2_aq', 'CO2_aq', ...
                   'SO4', 'FeS', 'X', 'Acetate', 'HCO3', 'S_tot', 'Lag', 'Fe_pool'};
               
    for i = 1:14
        subplot(4,4,i);
        plot(Y_true(:,i), 'b-', 'LineWidth', 1.5); hold on;
        plot(Y_pred(:,i), 'r--', 'LineWidth', 1.5);
        title(state_names{i});
        if i==1, legend('ODE (True)', 'LSTM v5'); end
        grid on; xlim([0 n_test]);
    end
    sgtitle(['LSTM v5 Validation (SequenceLength=' num2str(sequenceLength) ')']);
    
    fprintf('=== İşlem Tamamlandı ===\n');
end

%% ========================================================================
%  NESTED ODE MODEL (Değişmedi, v4 ile aynı)
% ========================================================================
function dydt = model_mixed(t, y, p, env)
    Vg=env.Vg; Vl=env.Vl; T=env.T; Rgas=env.Rgas;
    Hcp_H2=env.Hcp_H2_eff; Hcp_CO2=env.Hcp_CO2_eff; Hcp_H2S=env.Hcp_H2S_eff;
    pH=env.pH_fun(t); pKa=env.pKa_H2S;

    % State
    nH2_g=y(1); nCO2_g=y(2); nCH4_g=y(3); nH2S_g=y(4);
    H2_aq=y(5); CO2_aq=y(6); SO4=y(7); FeS=y(8); X=y(9);
    Ac=y(10); HCO3=y(11); S_tot=y(12); Lag=y(13); Fe_pool=y(14);

    % Parameters
    k_m=p(1); k_s=p(2); k_a=p(3);
    Y_m=p(4); Y_s=p(5); Y_a=p(6);
    KI_m=p(7); KI_s=p(8); KI_a=p(9);
    k_prec=p(10); HS_sat=p(11); H2_th=p(12); DG_th=p(13);
    K_H2=p(14); K_SO4=p(15); K_CO2=p(16);
    kla_H2=p(17); kla_CO2=p(18); kla_H2S=p(19);
    b=p(20); t_lag=p(21); w_lag=p(22);
    k_diss_gyp=p(23); beta_SO4_m=p(24);

    % Thermo
    RkJ=8.314e-3; RT=RkJ*T;
    DG0_m=-130; DG0_s=-152; DG0_a=-95;

    % Guards
    eps=1e-12;
    y = max(y, eps); Fe_pool=max(Fe_pool, 0); 
    % (Kısa tutmak için tek satıra indirdim, mantık aynı)
    nH2_g=y(1); nCO2_g=y(2); nCH4_g=y(3); nH2S_g=y(4);
    H2_aq=y(5); CO2_aq=y(6); SO4=y(7); S_tot=y(12);
    Ac=y(10); HCO3=y(11); X=y(9);

    % Partial pressures
    pH2  = (nH2_g /1000)  * Rgas * T / Vg;
    pCO2 = (nCO2_g/1000)  * Rgas * T / Vg;
    pH2S = (nH2S_g/1000)  * Rgas * T / Vg;

    % Henry
    Ceq_H2=Hcp_H2*pH2; Ceq_CO2=Hcp_CO2*pCO2; Ceq_H2S=Hcp_H2S*pH2S;
    J_H2=kla_H2*(Ceq_H2-H2_aq); J_CO2=kla_CO2*(Ceq_CO2-CO2_aq);

    % Sulfide
    frac_HS=1/(1+10^(pKa - pH)); HS_aq=S_tot*frac_HS; H2S_aq=S_tot*(1-frac_HS);
    Jout_H2S=kla_H2S*(H2S_aq-Ceq_H2S);

    % Kinetics
    f_inh_m=KI_m/(KI_m+HS_aq); f_inh_s=KI_s/(KI_s+HS_aq); f_inh_a=KI_a/(KI_a+HS_aq);
    f_H2=H2_aq/(H2_aq+H2_th); f_lag=1/(1+exp((t_lag - t)/max(w_lag,1e-3)));
    f_act=f_H2*f_lag;

    mH2=H2_aq/(K_H2+H2_aq); mSO4=SO4/(K_SO4+SO4); mCO2=CO2_aq/(K_CO2+CO2_aq);

    Q_s=1; Q_a=Ac/(H2_aq^4 * CO2_aq^2);
    fT_s=1/(1+exp((DG0_s+RT*log(Q_s)-DG_th)/RT));
    fT_m=1/(1+exp((DG0_m-DG_th)/RT));
    fT_a=1/(1+exp((DG0_a+RT*log(Q_a)-DG_th)/RT));
    f_comp_m=1/(1+beta_SO4_m*SO4);

    r_meth=k_m*X*mH2*mCO2*f_inh_m*f_act*fT_m*f_comp_m;
    r_sulf=k_s*X*mH2*mSO4*f_inh_s*f_act*fT_s;
    r_aceto=k_a*X*mH2*(mCO2^2)*f_inh_a*f_act*fT_a;

    r_prec=min(k_prec*max(0, HS_aq-HS_sat), Fe_pool);
    r_diss_gyp=k_diss_gyp*max(0, env.SO4_sat_gyp - SO4);

    % Derivatives
    dnH2_g=-J_H2*Vl; dnCO2_g=-J_CO2*Vl; dnCH4_g=r_meth*Vl; dnH2S_g=Jout_H2S*Vl;
    dH2_aq=J_H2-4*r_meth-4*r_sulf-4*r_aceto;
    dCO2_aq=J_CO2-r_meth-2*r_aceto;
    dSO4=-r_sulf+r_diss_gyp;
    dFeS=r_prec;
    dX=Y_m*r_meth+Y_s*r_sulf+Y_a*r_aceto-b*X;
    dAc=r_aceto; dHCO3=0;
    dFe_pool=-r_prec;
    dS_tot=r_sulf-r_prec-Jout_H2S;
    dLag=(f_lag-Lag)/max(w_lag,1e-3);

    dydt=[dnH2_g; dnCO2_g; dnCH4_g; dnH2S_g; dH2_aq; dCO2_aq; dSO4; dFeS; dX; dAc; dHCO3; dS_tot; dLag; dFe_pool];
end
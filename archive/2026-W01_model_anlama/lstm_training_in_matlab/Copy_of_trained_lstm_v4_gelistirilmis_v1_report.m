function lstm_train_v5_report
% ============================================================================
% LSTM EĞİTİMİ V5 - RAPORLAMA MODÜLÜ EKLENMİŞ VERSİYON
% ============================================================================
% YENİLİK:
% - Grafiklerin yanı sıra 'lstm_validation_results.txt' dosyası oluşturur.
% - Bu dosya sayesinde gerçek ve tahmin değerlerini yan yana görebilirsin.
% ============================================================================

    fprintf('=== LSTM Eğitimi ve Raporlama Başlıyor ===\n\n');

    %% 1. FİTTED PARAMETERS'I YÜKLE
    fprintf('[1/7] Parametreler yükleniyor...\n');
    
    % Dosya yolları (Kendi bilgisayarına göre ayarla)
    code_folder = 'd:\chemical_thesis_repo\2026-W01_model_anlama\code\matlab\';
    param_file = fullfile(code_folder, 'best_fit_params_Basalt_25C.mat');
    
    if ~isfile(param_file)
        % Dosya yoksa, scriptin olduğu yerde ara
        if isfile('best_fit_params_Basalt_25C.mat')
             param_file = 'best_fit_params_Basalt_25C.mat';
             code_folder = pwd;
        else
             error('Parametre dosyası bulunamadı! Önce ODE modelini çalıştır.');
        end
    end

    load(param_file, 'p_fit', 'env', 'y0');
    addpath(code_folder); 
    fprintf('   ✓ Parametreler hazır.\n');

    %% 2. ODE ÇÖZÜMÜ (VERİ ÜRETİMİ)
    fprintf('[2/7] Sentetik eğitim verisi üretiliyor (2500 nokta)...\n');

    t_end = 20; % Varsayılan süre (gün)
    t_train = linspace(0, t_end, 2500)'; 
    
    odes = @(t,y) model_mixed(t,y,p_fit,env);
    opts = odeset('NonNegative',1:14,'RelTol',1e-9,'AbsTol',1e-11,'MaxStep',0.1);

    [~, y_sim] = ode15s(odes, t_train, y0, opts);
    xTrain_raw = y_sim'; 

    %% 3. VERİ ÖN İŞLEME
    fprintf('[3/7] Normalizasyon yapılıyor...\n');

    log_indices = [4, 8, 10, 13, 14]; % Log transform uygulanacaklar
    xTrain = xTrain_raw;
    xTrain(log_indices, :) = log1p(xTrain_raw(log_indices, :));

    xTrain_mean = mean(xTrain, 2);
    xTrain_std  = std(xTrain, 0, 2);
    xTrain_std(xTrain_std < 1e-9) = 1;

    xTrain_norm = (xTrain - xTrain_mean) ./ xTrain_std;
    
    norm_params.mean = xTrain_mean;
    norm_params.std = xTrain_std;
    norm_params.log_indices = log_indices;

    %% 4. SEQUENCE HAZIRLIĞI
    fprintf('[4/7] Sequence verisi hazırlanıyor (Window=50)...\n');
    sequenceLength = 50; 
    
    X = {}; Y = [];
    num_samples = size(xTrain_norm, 2) - sequenceLength;
    
    for i = 1:num_samples
        X{end+1} = xTrain_norm(:, i : i+sequenceLength-1);
        Y(end+1, :) = xTrain_norm(:, i+sequenceLength)';
    end

    %% 5. MİMARİ VE EĞİTİM
    fprintf('[5/7] Model eğitiliyor (Stacked LSTM)...\n');
    
    numFeatures = size(xTrain_norm, 1);
    layers = [ ...
        sequenceInputLayer(numFeatures, 'Name', 'input')
        lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'lstm1')
        dropoutLayer(0.2)
        lstmLayer(64, 'OutputMode', 'last', 'Name', 'lstm2')
        dropoutLayer(0.2)
        fullyConnectedLayer(numFeatures)
        regressionLayer('Name', 'output')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 1000, 'MiniBatchSize', 64, ...
        'InitialLearnRate', 1e-3, 'GradientThreshold', 1, ...
        'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'training-progress');
    
    net = trainNetwork(X, Y, layers, options);

    %% 6. TAHMİN VE DENORMALİZASYON (GERÇEK DEĞERLERE DÖNÜŞ)
    fprintf('[6/7] Validasyon tahminleri yapılıyor...\n');
    
    n_test = min(200, length(X)); % İlk 200 adımı test et
    Y_pred_norm = predict(net, X(1:n_test));
    
    % Denormalizasyon (Norm -> Gerçek Skala)
    Y_pred = (Y_pred_norm .* norm_params.std') + norm_params.mean';
    Y_true = (Y(1:n_test, :) .* norm_params.std') + norm_params.mean';
    
    % Log-Inverse (Log1p -> Normal Sayı)
    Y_pred(:, log_indices) = expm1(Y_pred(:, log_indices));
    Y_true(:, log_indices) = expm1(Y_true(:, log_indices));
    
    state_names = {'nH2_g', 'nCO2_g', 'nCH4_g', 'nH2S_g', 'H2_aq', 'CO2_aq', ...
                   'SO4', 'FeS', 'X', 'Acetate', 'HCO3', 'S_tot', 'Lag', 'Fe_pool'};

    %% 7. RAPORLAMA (TXT DOSYASI OLUŞTURMA) - YENİ KISIM!
    fprintf('[7/7] RAPOR OLUŞTURULUYOR: lstm_validation_results.txt\n');
    
    % Kayıt yeri: scriptin olduğu klasör
    save_folder = fileparts(code_folder); 
    if isempty(save_folder), save_folder = pwd; end
    txt_filename = fullfile(save_folder, 'lstm_validation_results.txt');
    
    fid = fopen(txt_filename, 'w'); % Dosyayı yazma modunda aç
    
    fprintf(fid, '======================================================\n');
    fprintf(fid, '        LSTM MODEL VALİDASYON RAPORU (v5)            \n');
    fprintf(fid, '======================================================\n');
    fprintf(fid, 'Tarih: %s\n', datestr(now));
    fprintf(fid, 'Sequence Length: %d\n', sequenceLength);
    fprintf(fid, 'Test Edilen Adım Sayısı: %d\n\n', n_test);
    
    % --- BÖLÜM 1: GENEL HATA ÖZETİ (RMSE) ---
    fprintf(fid, '--- BÖLÜM 1: GENEL PERFORMANS (RMSE) ---\n');
    fprintf(fid, '(Değer ne kadar 0''a yakınsa o kadar iyi)\n\n');
    fprintf(fid, '%-15s | %-15s\n', 'DEĞİŞKEN', 'ORTALAMA HATA');
    fprintf(fid, '%s\n', repmat('-', 1, 35));
    
    rmse_vals = sqrt(mean((Y_true - Y_pred).^2));
    for i = 1:numFeatures
        fprintf(fid, '%-15s | %.6f\n', state_names{i}, rmse_vals(i));
    end
    fprintf(fid, '\n\n');
    
    % --- BÖLÜM 2: ADIM ADIM DETAYLI VERİ ---
    fprintf(fid, '--- BÖLÜM 2: ADIM ADIM KIYASLAMA TABLOSU ---\n');
    fprintf(fid, 'Burada her değişken için Gerçek (ODE) ve Tahmin (LSTM) değerlerini görebilirsiniz.\n');
    
    for i = 1:numFeatures
        fprintf(fid, '\n\n>>> DEĞİŞKEN: %s <<<\n', state_names{i});
        fprintf(fid, '%s\n', repmat('=', 1, 65));
        fprintf(fid, '%-10s | %-15s | %-15s | %-15s\n', 'Adım (t)', 'GERÇEK (Mavi)', 'TAHMİN (Kırmızı)', 'FARK (Hata)');
        fprintf(fid, '%s\n', repmat('-', 1, 65));
        
        for t = 1:n_test
            val_true = Y_true(t, i);
            val_pred = Y_pred(t, i);
            diff = val_true - val_pred;
            
            % Sayıları hizalı yazdır
            fprintf(fid, '%-10d | %15.6f | %15.6f | %15.6f\n', ...
                    t, val_true, val_pred, diff);
        end
    end
    
    fclose(fid); % Dosyayı kapat
    
    fprintf('   ✓ Rapor başarıyla oluşturuldu: %s\n', txt_filename);
    fprintf('=== İşlem Tamamlandı ===\n');
end

% === NESTED ODE FUNCTION (Değişmedi) ===
function dydt = model_mixed(t, y, p, env)
    Vg=env.Vg; Vl=env.Vl; T=env.T; Rgas=env.Rgas;
    Hcp_H2=env.Hcp_H2_eff; Hcp_CO2=env.Hcp_CO2_eff; Hcp_H2S=env.Hcp_H2S_eff;
    pH=env.pH_fun(t); pKa=env.pKa_H2S;
    
    nH2_g=y(1); nCO2_g=y(2); nCH4_g=y(3); nH2S_g=y(4);
    H2_aq=y(5); CO2_aq=y(6); SO4=y(7); FeS=y(8); X=y(9);
    Ac=y(10); HCO3=y(11); S_tot=y(12); Lag=y(13); Fe_pool=y(14);

    k_m=p(1); k_s=p(2); k_a=p(3);
    Y_m=p(4); Y_s=p(5); Y_a=p(6);
    KI_m=p(7); KI_s=p(8); KI_a=p(9);
    k_prec=p(10); HS_sat=p(11); H2_th=p(12); DG_th=p(13);
    K_H2=p(14); K_SO4=p(15); K_CO2=p(16);
    kla_H2=p(17); kla_CO2=p(18); kla_H2S=p(19);
    b=p(20); t_lag=p(21); w_lag=p(22);
    k_diss_gyp=p(23); beta_SO4_m=p(24);

    RkJ=8.314e-3; RT=RkJ*T;
    DG0_m=-130; DG0_s=-152; DG0_a=-95;
    
    y = max(y, 1e-12); Fe_pool=max(Fe_pool, 0); 
    nH2_g=y(1); nCO2_g=y(2); nCH4_g=y(3); nH2S_g=y(4);
    H2_aq=y(5); CO2_aq=y(6); SO4=y(7); S_tot=y(12);
    Ac=y(10); HCO3=y(11); X=y(9);

    pH2=(nH2_g/1000)*Rgas*T/Vg; pCO2=(nCO2_g/1000)*Rgas*T/Vg; pH2S=(nH2S_g/1000)*Rgas*T/Vg;
    Ceq_H2=Hcp_H2*pH2; Ceq_CO2=Hcp_CO2*pCO2; Ceq_H2S=Hcp_H2S*pH2S;
    J_H2=kla_H2*(Ceq_H2-H2_aq); J_CO2=kla_CO2*(Ceq_CO2-CO2_aq);

    frac_HS=1/(1+10^(pKa - pH)); HS_aq=S_tot*frac_HS; H2S_aq=S_tot*(1-frac_HS);
    Jout_H2S=kla_H2S*(H2S_aq-Ceq_H2S);

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
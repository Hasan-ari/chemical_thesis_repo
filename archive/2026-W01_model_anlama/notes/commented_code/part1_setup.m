% ============================================================================
% PART 1: SETUP - Sabitler, Veri Yükleme, Başlangıç Şartları
% ============================================================================
% Bu kısım kodun temel ayarlarını yapar:
% - Deney ortamı sabitleri (hacimler, sıcaklık)
% - Henry sabitleri (gaz-sıvı dengeleri için)
% - Deneysel veri okuma
% - pH interpolasyonu
% - Başlangıç konsantrasyonları
% ============================================================================

function anaerobic_model_two_phase_mixedSR_25C_v4
% AÇIKLAMA: Ana MATLAB fonksiyonu - tüm kod bu fonksiyonun içinde çalışır
% INPUT: Yok (dışarıdan parametre almaz)
% OUTPUT: Grafik pencereleri ve .dat dosyası oluşturur

% Sandstone @ 25 °C with effective Henry scale factors, H2S flux patch,
% and **finite Fe pool** limiting FeS precipitation.
% - T = 25 °C (298.15 K)
% - Hscp (c/p) in mmol/L/atm; partial pressures in atm
% - Writes reaction rates to .dat; plots species and rates

% ============================================================================
% 1. SABİTLER - Deney Ortamı Fiziksel Özellikleri
% ============================================================================

Vg = 0.14;   % headspace volume [L] ~ 140 mL
             % AÇIKLAMA: Gaz fazının hacmi (reaktörün üst kısmı, hava boşluğu)
             % DEĞİŞKEN TİPİ: Skaler sayı (tek bir değer)

Vl = 0.015;  % liquid volume   [L] ~ 15 mL
             % AÇIKLAMA: Sıvı fazının hacmi (reaktörün alt kısmı, su)
             % DEĞİŞKEN TİPİ: Skaler sayı

T  = 298.15; % K (25°C)
             % AÇIKLAMA: Sıcaklık Kelvin cinsinden (25°C = 298.15 K)
             % NEDEN KELVIN: İdeal gaz denklemi (PV=nRT) Kelvin gerektirir

R_gas = 0.082057; % L·atm/(mol·K)
                  % AÇIKLAMA: İdeal gaz sabiti R
                  % BİRİM: Litre * atmosfer / (mol * Kelvin)
                  % NEDEN BU DEĞER: P = nRT/V denkleminde kullanılır

% ============================================================================
% 2. HENRY SABİTLERİ - Gaz-Sıvı Dengesi İçin
% ============================================================================
% Henry Yasası: C_eq = H_cp * P
% Burada C_eq = denge konsantrasyonu (mmol/L), P = basınç (atm)
% H_cp = Henry sabiti (mmol/L/atm) şeklinde

%%% (Sandstone): use 25 °C base constants + scale factors (solubility variant Hscp=c/p)
Hcp_H2_base  = 0.78;  % H2  (25 °C) (mmol/L/atm)
                      % AÇIKLAMA: H2 için Henry sabiti
                      % ANLAMLI: Düşük değer = az çözünür (H2 suda az çözünür)

Hcp_CO2_base = 34.0;  % CO2 (25 °C) (mmol/L/atm)
                      % AÇIKLAMA: CO2 için Henry sabiti
                      % ANLAMLI: H2'den 43 kat daha çözünür (34.0 vs 0.78)

Hcp_H2S_base = 90.0;  % H2S (25 °C) (mmol/L/atm)
                      % AÇIKLAMA: H2S için Henry sabiti
                      % ANLAMLI: En çok çözünen gaz (90.0 >> 34.0 >> 0.78)

% ============================================================================
% 3. ÖLÇEKLEME FAKTÖRLERİ (Scale Factors)
% ============================================================================
% AÇIKLAMA: Henry sabitlerini ince ayar yapmak için çarpanlar
% BAŞLANGIÇ DEĞER: 1.0 (değişiklik yok demektir)
% FİTTİNG SIRASINDA: Bu değerler optimize edilebilir (0.85-1.15 arası)

% Optional scale factors (keep 1.0 unless you need fine-tuning)
phi_H2  = 1.00;  % H2 çarpanı
                 % AÇIKLAMA: Hcp_H2_base * phi_H2 = efektif Henry sabiti

phi_CO2 = 1.00;  % CO2 çarpanı
                 % AÇIKLAMA: Hcp_CO2_base * phi_CO2 = efektif Henry sabiti

% (You can add phi_H2S if you need it; we keep H2S as base here)

% ============================================================================
% 4. EFEKTİF HENRY SABİTLERİ - Gerçekte Kullanılanlar
% ============================================================================
% FORMÜL: Hcp_eff = phi * Hcp_base
% AÇIKLAMA: Bu değerler tüm kod boyunca kullanılır

% Effective Henry constants used by the model
Hcp_H2_eff  = phi_H2  * Hcp_H2_base;   % = 1.0 * 0.78 = 0.78 mmol/L/atm
Hcp_CO2_eff = phi_CO2 * Hcp_CO2_base;  % = 1.0 * 34.0 = 34.0 mmol/L/atm
Hcp_H2S_eff = Hcp_H2S_base;            % = 90.0 mmol/L/atm (phi yok)

% ============================================================================
% 5. DENEYSEL VERİ YÜKLEME
% ============================================================================
% AÇIKLAMA: Deneyde ölçülen değerleri txt dosyasından okur
% readmatrix: MATLAB fonksiyonu, dosyayı matris olarak yükler

% -------------------- Load experimental data --------------------
raw = readmatrix('Muller_2024_H2_Sandstone_at_25C.txt');
% AÇIKLAMA: txt dosyasını okur
% SONUÇ: raw = [N satır x 7 sütun] matris
% SÜTUNLAR: [zaman, nH2_g, nCO2_g, nCH4_g, nH2S_g, pH, SO4]

% ============================================================================
% 6. VERİ EKSTRAKSİYONU - Matrisin Sütunlarını Ayır
% ============================================================================

t_exp       = raw(:,1);  % days
                         % AÇIKLAMA: raw matrisinin TÜM satırları (:), 1. sütunu
                         % ÖRNEK: [0; 0.5; 1.0; 1.5; ...] şeklinde bir vektör
                         % BİRİM: gün

nH2_g_exp   = raw(:,2) / 1000; % µmol -> mmol
                               % AÇIKLAMA: 2. sütunu al ve 1000'e böl
                               % NEDEN BÖL: µmol'ü mmol'e çevirmek için
                               % BİRİM DÖNÜŞÜMÜ: 1 mmol = 1000 µmol

nCO2_g_exp  = raw(:,3) / 1000; % µmol -> mmol
                               % AÇIKLAMA: Aynı şekilde CO2 verisi

nCH4_g_exp  = raw(:,4) / 1000; % µmol -> mmol
                               % AÇIKLAMA: Aynı şekilde CH4 verisi

nH2S_g_exp  = raw(:,5) / 1000; % µmol -> mmol
                               % AÇIKLAMA: Aynı şekilde H2S verisi

pH_exp      = raw(:,6);        % pH
                               % AÇIKLAMA: pH verisi (boyutsuz, 0-14 arası)
                               % BİRİM DÖNÜŞÜMÜ YOK: pH zaten doğru birimde

SO4_exp     = raw(:,7);        % sulfate mM (mmol/L)
                               % AÇIKLAMA: Sülfat konsantrasyonu
                               % BİRİM: mM = mmol/L (milimolar)

% ============================================================================
% 7. VERİ MATRİSİ OLUŞTURMA - Fitting İçin
% ============================================================================

data_exp    = [nH2_g_exp, nCO2_g_exp, nCH4_g_exp, nH2S_g_exp, SO4_exp];
% AÇIKLAMA: 5 sütunlu matris oluştur (gaz verileri + SO4)
% BOYUT: [N satır x 5 sütun]
% NEDEN: lsqnonlin optimizasyonu bu matrise ihtiyaç duyacak
% İÇERİK: Model çıktıları ile karşılaştırılacak deneysel veriler

% ============================================================================
% 8. pH İNTERPOLASYONU - Zamana Bağlı pH Fonksiyonu
% ============================================================================

% pH interpolant
pH_fun = @(t) max(0, interp1(t_exp, pH_exp, t, 'linear', 'extrap'));
% AÇIKLAMA: Anonim fonksiyon oluştur (lambda function gibi)
% YAPISI: @(input) fonksiyon_gövdesi
% AMAÇ: Herhangi bir t zamanı için pH değeri döndür

% İÇ YAPISI:
%   - interp1(t_exp, pH_exp, t, 'linear', 'extrap')
%     * t_exp: Bilinen zaman noktaları (x ekseni)
%     * pH_exp: Bilinen pH değerleri (y ekseni)
%     * t: İstenen zaman noktası (sorgulanacak x)
%     * 'linear': Doğrusal interpolasyon kullan
%     * 'extrap': Sınırların dışında ekstrapolasyon yap
%   - max(0, ...): Sonucu 0'dan küçük olma, minimum 0 döndür
%     * NEDEN: Negatif pH fiziksel olarak anlamsız

% KULLANIM ÖRNEĞİ:
%   pH_fun(2.5)  -> 2.5 gün zamanındaki pH'ı döndürür (örn: 7.35)
%   pH_fun(100)  -> 100 gün zamanındaki pH'ı döndürür (ekstrapolasyon ile)

% ============================================================================
% 9. BAŞLANGIÇ GAZ BASINÇLARI - İdeal Gaz Denklemi
% ============================================================================
% FORMÜL: P = (n/1000) * R * T / V
%   n: mmol cinsinden mol sayısı
%   1000: mmol -> mol dönüşümü
%   R: Gaz sabiti (0.082057 L·atm/(mol·K))
%   T: Sıcaklık (298.15 K)
%   V: Hacim (0.14 L = Vg)

% -------------------- Initial aqueous equilibrium --------------------
pH2  = (nH2_g_exp(1)/1000)  * R_gas * T / Vg; % atm
% AÇIKLAMA: t=0 zamanındaki H2 kısmi basıncı
% ADIM ADIM:
%   1. nH2_g_exp(1): İlk zaman noktasındaki H2 miktarı (mmol)
%   2. /1000: mmol -> mol dönüşümü
%   3. * R_gas * T: İdeal gaz denklemi PV=nRT'den P = nRT/V
%   4. / Vg: Hacme böl
% SONUÇ: atm cinsinden basınç (örn: 0.05 atm)

pCO2 = (nCO2_g_exp(1)/1000) * R_gas * T / Vg; % atm
% AÇIKLAMA: Aynı mantık, CO2 için

pH2S = (nH2S_g_exp(1)/1000) * R_gas * T / Vg; % atm
% AÇIKLAMA: Aynı mantık, H2S için

% ============================================================================
% 10. BAŞLANGIÇ SIVI KONSANTRASYONLARİ - Henry Yasası
% ============================================================================
% FORMÜL: C_eq = H_cp * P
%   C_eq: Dengede sıvıdaki konsantrasyon (mmol/L)
%   H_cp: Henry sabiti (mmol/L/atm)
%   P: Gaz basıncı (atm)

H2_aq0  = Hcp_H2_eff  * pH2;   % mmol/L
% AÇIKLAMA: t=0'daki sıvı fazında çözünmüş H2 konsantrasyonu
% HESAP: 0.78 (mmol/L/atm) * pH2 (atm) = mmol/L
% ÖRNEK: Eğer pH2=0.05 atm ise -> 0.78*0.05 = 0.039 mmol/L

CO2_aq0 = Hcp_CO2_eff * pCO2;  % mmol/L
% AÇIKLAMA: t=0'daki sıvı fazında çözünmüş CO2 konsantrasyonu

% ============================================================================
% 11. DİAGNOSTİK BASINÇ HESABI - Toplam Basınç
% ============================================================================

% Diagnostics
Ptot_0 = ((nH2_g_exp(1)+nCO2_g_exp(1)+nCH4_g_exp(1)+nH2S_g_exp(1))/1000) * R_gas * T / Vg;
% AÇIKLAMA: Tüm gazların toplam kısmi basıncı (Dalton Yasası)
% FORMÜL: P_total = P_H2 + P_CO2 + P_CH4 + P_H2S
% ADIM ADIM:
%   1. nH2_g_exp(1)+nCO2_g_exp(1)+... : Tüm gaz mollerini topla (mmol)
%   2. /1000: mmol -> mol
%   3. * R_gas * T / Vg: İdeal gaz denklemi P=nRT/V
% SONUÇ: Toplam basınç (atm), örn: 1.2 atm

% ============================================================================
% 12. EKRANA YAZDIRMA - Diagnostics
% ============================================================================

fprintf('\n[Henry/Pressure @ t0] P_tot=%.3f atm \n p_H2=%.3f, p_CO2=%.3f, p_H2S=%.4f atm \n Ceq: H2=%.4f, CO2=%.3f mmol/L\n', ...
        Ptot_0, pH2, pCO2, pH2S, H2_aq0, CO2_aq0);
% AÇIKLAMA: fprintf = "formatted print" (C'deki printf gibi)
% YAPISI:
%   fprintf('format stringi', değişken1, değişken2, ...)
%   %.3f: 3 ondalık basamak göster (float)
%   %.4f: 4 ondalık basamak göster
%   \n: Yeni satır (new line)
% ÖRNEK ÇIKTI:
%   [Henry/Pressure @ t0] P_tot=1.234 atm
%    p_H2=0.050, p_CO2=0.030, p_H2S=0.0001 atm
%    Ceq: H2=0.0390, CO2=1.020 mmol/L

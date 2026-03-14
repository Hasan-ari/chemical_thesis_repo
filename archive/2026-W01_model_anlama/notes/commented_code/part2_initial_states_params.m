% ============================================================================
% PART 2: INITIAL STATES + PARAMETERS
% ============================================================================
% Bu kısım:
% - y0: Başlangıç durum vektörü (14 state variable)
% - p0, lb, ub: Parametre tahmin değerleri ve sınırları
% - fit_opts: Optimizasyon ayarları
% - env: Çevre değişkenleri struct'ı
% ============================================================================

%% -------------------- Initial states (14 species with Fe pool) --------------------
% STATE VEKTÖRÜ (y): ODE solver'ın çözeceği 14 değişken
% AÇIKLAMA: Her bir satır bir state variable'ı temsil eder
% SIRA ÖNEMLİ: model_mixed() fonksiyonu bu sırayı bilmeli!

% [ nH2_g, nCO2_g, nCH4_g, nH2S_g, H2(aq), CO2(aq), SO4, FeS, X, Acetate, HCO3, S_tot, Lag, Fe_pool ]
%   y(1)   y(2)    y(3)    y(4)     y(5)     y(6)    y(7) y(8) y(9) y(10)   y(11)  y(12)  y(13) y(14)

S_tot0   = 1;  % tiny sulfide seed to allow early H2S(g) appearance
               % AÇIKLAMA: Başlangıçta biraz sülfür koy (mmol/L)
               % NEDEN: Eğer 0 olursa H2S gazı oluşamaz (bölme hatasına yol açabilir)
               % BİRİM: mmol/L

%%% PATCH (Fe pool): initial dissolved Fe(II) pool (mmol/L)
Fe_pool0 = 0.10;  % choose 0.05–0.5 mM based on medium; can fit if needed
                  % AÇIKLAMA: Başlangıçta çözünmüş demir miktarı (Fe²⁺)
                  % KULLANIM: FeS çökelmesi için demir gerekli
                  % TİP DEĞER: 0.05-0.5 mmol/L arası (deneysel ortama göre)
                  % BİRİM: mmol/L

% ============================================================================
% y0 VEKTÖRÜ OLUŞTURMA - 14 Elemanlı Başlangıç Durumu
% ============================================================================

y0 = [ nH2_g_exp(1), nCO2_g_exp(1), nCH4_g_exp(1), nH2S_g_exp(1), ...
       H2_aq0, CO2_aq0, SO4_exp(1), ...
       0.01, 0.01, 0, 0, S_tot0, 0, ...
       Fe_pool0 ];

% SATIR SATIR AÇIKLAMA:
% SATIR 1: nH2_g_exp(1), nCO2_g_exp(1), nCH4_g_exp(1), nH2S_g_exp(1)
%   -> y(1), y(2), y(3), y(4): Gaz fazı başlangıç değerleri (mmol)
%   -> Deneysel veriden alınan ilk zaman noktası değerleri

% SATIR 2: H2_aq0, CO2_aq0, SO4_exp(1)
%   -> y(5), y(6), y(7): Sıvı fazı başlangıç konsantrasyonları
%   -> H2_aq0: Henry yasasından hesaplanan H2 (mmol/L)
%   -> CO2_aq0: Henry yasasından hesaplanan CO2 (mmol/L)
%   -> SO4_exp(1): Deneysel veriден sülfat (mmol/L)

% SATIR 3: 0.01, 0.01, 0, 0, S_tot0, 0
%   -> y(8)=0.01: FeS başlangıcı (az miktarda, mmol/L)
%   -> y(9)=0.01: Biomass başlangıcı (küçük seed, mmol/L)
%   -> y(10)=0: Acetate başlangıcı (yok, mmol/L)
%   -> y(11)=0: HCO3 başlangıcı (sabit, mmol/L)
%   -> y(12)=S_tot0: Toplam çözünmüş sülfür (1 mmol/L)
%   -> y(13)=0: Lag variable başlangıcı (0 = henüz aktive olmamış)

% SATIR 4: Fe_pool0
%   -> y(14)=0.10: Çözünmüş demir havuzu (mmol/L)

% VEKTÖR BOYUTU: [1x14] (1 satır, 14 sütun)
% VEKTÖR TİPİ: Satır vektörü (MATLAB'da ode15s satır vektör bekler)

%% -------------------- Parameter vector --------------------
% PARAMETRE VEKTÖRÜ (p): 28 adet parametre
% Bu parametreler OPTİMİZASYON sırasında ayarlanacak
% Her parametrenin bir ALT SINIR (lb) ve ÜST SINIR (ub) var

% -------------------- Parameters to fit --------------------
% p = [k_m, k_s, k_a, Y_m, Y_s, Y_a, KI_m, KI_s, KI_a, k_prec, HS_sat, H2_th, DG_th, ...
%      K_H2, K_SO4, K_CO2, kla_H2, kla_CO2, kla_H2S, b, t_lag, w_lag, k_diss_gyp, beta_SO4_m, ...
%      phi_H2, phi_CO2, phi_H2S, alpha_H2S]

% ============================================================================
% p0: BAŞLANGIÇ TAHMİNİ (Initial Guess)
% ============================================================================
% AÇIKLAMA: Optimizasyon bu değerlerden başlayacak
% STRATEJİ: Makul değerler seç, yoksa çözüm bulamaz

p0 = [ 0.06, 0.08, 0.03, ...        % k_m, k_s, k_a (rate constants, 1/day)
       0.06, 0.05, 0.05, ...        % Y_m, Y_s, Y_a (yield coefficients, mmolX/mmol)
       0.20, 0.20, 0.20, ...        % KI_m, KI_s, KI_a (inhibition constants, mmol/L)
       0.02, 0.10, 0.02, -12, ...   % k_prec, HS_sat, H2_th, DG_th
       0.50, 0.50, 0.80, ...        % K_H2, K_SO4, K_CO2 (Monod half-sat, mmol/L)
       10.0, 10.0, 25.0, ...        % kla_H2, kla_CO2, kla_H2S (mass transfer, 1/day)
       0.01, 3.0, 0.7, ...          % b, t_lag, w_lag
       0.12, 0.10, ...              % k_diss_gyp, beta_SO4_m
       1.00, 1.00, 1.00, ...        % phi_H2, phi_CO2, phi_H2S
       1.00 ];                      % alpha_H2S

% SATIR SATIR AÇIKLAMA:
% ----------------------
% SATIR 1: k_m=0.06, k_s=0.08, k_a=0.03
%   -> p(1), p(2), p(3): Maksimum reaksiyon hızı sabitleri
%   -> BİRİM: 1/gün (per day)
%   -> ANLAMLI: k_s > k_m > k_a (sülfat indirgeme > metan > asetat)

% SATIR 2: Y_m=0.06, Y_s=0.05, Y_a=0.05
%   -> p(4), p(5), p(6): Biyokütle verimi (yield)
%   -> BİRİM: mmol biomass / mmol substrate
%   -> ANLAMLI: 0.05-0.06 tipik değerler (%5-6 dönüşüm)

% SATIR 3: KI_m=0.20, KI_s=0.20, KI_a=0.20
%   -> p(7), p(8), p(9): Sülfür inhibisyon sabitleri
%   -> BİRİM: mmol/L (HS⁻ konsantrasyonu)
%   -> ANLAMLI: Düşük KI = güçlü inhibisyon

% SATIR 4: k_prec=0.02, HS_sat=0.10, H2_th=0.02, DG_th=-12
%   -> p(10): k_prec = FeS çökelme hızı (1/day)
%   -> p(11): HS_sat = Çökelme için eşik HS⁻ (mmol/L)
%   -> p(12): H2_th = Aktivasyon için eşik H2 (mmol/L)
%   -> p(13): DG_th = Termodinamik eşik enerji (kJ/mol)

% SATIR 5: K_H2=0.50, K_SO4=0.50, K_CO2=0.80
%   -> p(14), p(15), p(16): Monod yarı doygunluk sabitleri
%   -> BİRİM: mmol/L
%   -> ANLAMLI: m_X = X/(K_X + X) denkleminde kullanılır
%   -> YORUM: Düşük K = yüksek afinite (substratı kolay kullanır)

% SATIR 6: kla_H2=10.0, kla_CO2=10.0, kla_H2S=25.0
%   -> p(17), p(18), p(19): Gaz-sıvı kütle transfer katsayıları
%   -> BİRİM: 1/gün
%   -> ANLAMLI: Yüksek kla = hızlı transfer (gaz-sıvı dengesi hızlı)

% SATIR 7: b=0.01, t_lag=3.0, w_lag=0.7
%   -> p(20): b = Biyokütle bozunma katsayısı (1/day)
%   -> p(21): t_lag = Lag fazı merkez zamanı (gün)
%   -> p(22): w_lag = Lag fazı genişliği (gün)
%   -> ANLAMLI: 3 gün civarında aktivasyon başlar, 0.7 gün geçiş süresi

% SATIR 8: k_diss_gyp=0.12, beta_SO4_m=0.10
%   -> p(23): k_diss_gyp = Gypsum çözünme hızı (1/day)
%   -> p(24): beta_SO4_m = Sülfat-metan rekabet katsayısı (1/mM)

% SATIR 9: phi_H2=1.00, phi_CO2=1.00, phi_H2S=1.00
%   -> p(25), p(26), p(27): Henry sabiti düzeltme faktörleri
%   -> BİRİM: boyutsuz (1.0 = değişiklik yok)
%   -> FİTTİNG: 0.85-1.15 arası optimize edilebilir

% SATIR 10: alpha_H2S=1.00
%   -> p(28): H2S gaz transferi ekstra çarpanı
%   -> BİRİM: boyutsuz
%   -> ANLAMLI: kla_H2S * alpha_H2S = efektif transfer

% VEKTÖR BOYUTU: [1x28] (1 satır, 28 sütun)

% ============================================================================
% lb: ALT SINIR (Lower Bound)
% ============================================================================
% AÇIKLAMA: Her parametrenin alabileceği EN KÜÇÜK değer
% AMAÇ: Optimizasyon bu sınırların DIŞINA çıkamaz
% FİZİKSEL ANLAM: Negatif hız sabiti, negatif hacim gibi saçma değerleri engelle

lb = [ 1e-4, 1e-4, 1e-4, ...        % k_m, k_s, k_a ≥ 0.0001
       0.01, 0.01, 0.01, ...        % Y_m, Y_s, Y_a ≥ 0.01
       1e-3, 1e-3, 1e-3, ...        % KI_m, KI_s, KI_a ≥ 0.001
       0.0, 0.0, 0.0, -50, ...      % k_prec≥0, HS_sat≥0, H2_th≥0, DG_th≥-50
       1e-3, 1e-3, 1e-3, ...        % K_H2, K_SO4, K_CO2 ≥ 0.001
       0.1, 0.1, 0.1, ...           % kla_H2, kla_CO2, kla_H2S ≥ 0.1
       0, 0, 0.1, ...               % b≥0, t_lag≥0, w_lag≥0.1
       0.01, 0.00, ...              % k_diss_gyp≥0.01, beta_SO4_m≥0
       0.85, 0.85, 0.90, ...        % phi_H2≥0.85, phi_CO2≥0.85, phi_H2S≥0.90
       0.70];                       % alpha_H2S≥0.70

% ÖRNEKLER:
%   lb(1) = 1e-4 -> k_m en az 0.0001 olabilir (çok küçük ama 0 değil)
%   lb(13) = -50 -> DG_th en az -50 kJ/mol olabilir (çok negatif ama sınırlı)
%   lb(25) = 0.85 -> phi_H2 en az 0.85 olabilir (1.0'dan %15 azalabilir)

% ============================================================================
% ub: ÜST SINIR (Upper Bound)
% ============================================================================
% AÇIKLAMA: Her parametrenin alabileceği EN BÜYÜK değer

ub = [ 5, 5, 5, ...                 % k_m, k_s, k_a ≤ 5
       0.5, 0.5, 0.5, ...           % Y_m, Y_s, Y_a ≤ 0.5
       5, 5, 5, ...                 % KI_m, KI_s, KI_a ≤ 5
       1.0, 5.0, 1.0, 0, ...        % k_prec≤1, HS_sat≤5, H2_th≤1, DG_th≤0
       20, 20, 20, ...              % K_H2, K_SO4, K_CO2 ≤ 20
       200, 200, 200, ...           % kla_H2, kla_CO2, kla_H2S ≤ 200
       0.2, 10, 2.0, ...            % b≤0.2, t_lag≤10, w_lag≤2.0
       2.00, 1.00, ...              % k_diss_gyp≤2, beta_SO4_m≤1
       1.15, 1.15, 1.10, ...        % phi_H2≤1.15, phi_CO2≤1.15, phi_H2S≤1.10
       3.00 ];                      % alpha_H2S≤3.00

% ÖRNEKLER:
%   ub(1) = 5 -> k_m en fazla 5 olabilir
%   ub(13) = 0 -> DG_th en fazla 0 olabilir (negatif veya 0)
%   ub(25) = 1.15 -> phi_H2 en fazla 1.15 olabilir (1.0'dan %15 artabilir)

% VEKTÖR BOYUTLARI:
%   p0: [1x28]
%   lb: [1x28]
%   ub: [1x28]
% HEPSİ AYNI BOYUTTA OLMALI!

% ============================================================================
% fit_opts: OPTİMİZASYON AYARLARI
% ============================================================================

fit_opts = optimoptions('lsqnonlin','Display','iter','MaxFunctionEvaluations',6000);
% AÇIKLAMA: lsqnonlin optimizer için seçenekler oluştur
% YAPISI: optimoptions('optimizer_adı', 'seçenek1', değer1, 'seçenek2', değer2, ...)

% SEÇENEKLERİN AÇIKLAMASI:
% -------------------------
% 'lsqnonlin': Least-Squares Nonlinear optimizer
%   -> AMAÇ: Minimize et: sum((model - data)^2)
%   -> MATLAB'ın built-in fonksiyonu

% 'Display','iter': Her iterasyonda bilgi yazdır
%   -> 'iter': Her adımda ekrana yazdır
%   -> Alternatifler: 'off' (hiç), 'final' (sadece son), 'notify' (sorun varsa)

% 'MaxFunctionEvaluations',6000: En fazla 6000 fonksiyon çağrısı yap
%   -> AMAÇ: Sonsuz döngüye girmemek için limit koy
%   -> Default: 100*parametre_sayısı = 100*28 = 2800
%   -> 6000 seçilmiş: Daha fazla iterasyon için (daha iyi sonuç)

% DİĞER OLASI SEÇENEKLER (kullanılmamış):
%   'TolFun', 1e-6: Fonksiyon toleransı (ne kadar küçük değişimde dursun)
%   'TolX', 1e-6: Parametre toleransı (parametreler ne kadar az değişince dursun)
%   'Algorithm', 'trust-region-reflective': Algoritma seçimi

%% -------------------- Environment pack --------------------
% ENV STRUCT: Çevre değişkenlerini bir struct içinde topla
% AMAÇ: Fonksiyonlara tek bir parametre olarak geç (env)
% STRUCT YAPISI: env.alan_adı = değer

env.Vg = Vg;
% AÇIKLAMA: env struct'ına Vg alanı ekle, değerini Vg yap
% KULLANIM: model_mixed() içinde env.Vg diye erişilir

env.Vl = Vl;
% AÇIKLAMA: Sıvı hacmini struct'a ekle

env.T = T;
% AÇIKLAMA: Sıcaklığı struct'a ekle

env.Rgas = R_gas;
% AÇIKLAMA: Gaz sabitini struct'a ekle

env.Hcp_H2_eff  = Hcp_H2_eff;
% AÇIKLAMA: Efektif Henry sabitlerini struct'a ekle
env.Hcp_CO2_eff = Hcp_CO2_eff;
env.Hcp_H2S_eff = Hcp_H2S_eff;

env.pH_fun      = pH_fun;
% AÇIKLAMA: pH interpolasyon FONKSİYONUNU struct'a ekle
% ÖNEMLİ: Bu bir fonksiyon handle (@(t) ...), sayı değil!
% KULLANIM: env.pH_fun(2.5) -> 2.5 gün zamanındaki pH

env.pKa_H2S     = 7.05;  % modest T-adjusted pKa1; tweak only if needed
% AÇIKLAMA: H2S'nin pKa değeri (asit denge sabiti)
% FİZİKSEL ANLAM: pH=pKa'da H2S ve HS⁻ eşit miktarda
% BİRİM: boyutsuz (log ölçeği)

% PATCH equilibrium SO4 at 25 °C ~30–40 mM (was 15.0 mM)
env.SO4_sat_gyp = 15.0; % mM — matches experimental plateaus
% AÇIKLAMA: Doygun sülfat konsantrasyonu (Gypsum çözünürlüğünden)
% BİRİM: mM = mmol/L
% KULLANIM: r_diss_gyp hesabında kullanılır

% STRUCT ÖZETİ:
% env = struct ile 9 alan içerir:
%   .Vg, .Vl, .T, .Rgas (fiziksel sabitler)
%   .Hcp_H2_eff, .Hcp_CO2_eff, .Hcp_H2S_eff (Henry sabitleri)
%   .pH_fun (fonksiyon handle)
%   .pKa_H2S, .SO4_sat_gyp (kimyasal sabitler)

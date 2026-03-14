% ============================================================================
% PART 7: HELPER FUNCTIONS - Yardımcı Fonksiyonlar
% ============================================================================
% Bu kısım 3 yardımcı fonksiyon içerir:
% 1. rate_out_mixed: Reaksiyon hızlarını hesapla (plotting için)
% 2. speciate_sulfide: Sülfür speslerini ayır (H2S vs HS⁻)
% 3. rmse_equal: RMSE metriği hesapla
% ============================================================================

%% -------------------- Reaction rates (for plotting + .dat) --------------------
function dr = rate_out_mixed(t, y, p, env)
% ============================================================================
% FONKSİYON: RATE_OUT_MIXED
# ============================================================================
# AMAÇ: Belirli bir (t, y) noktasında reaksiyon hızlarını hesapla
%        (model_mixed içindeki hız hesaplarının kopyası)
%
% INPUT:
%   t   : Zaman (scalar, gün)
%   y   : State vektörü [14x1]
%   p   : Parametre vektörü [28x1]
%   env : Çevre değişkenleri struct
%
% OUTPUT:
%   dr  : Hızlar vektörü [1x4]
%         dr = [r_meth, r_sulf, r_prec, r_aceto]

% ============================================================================
% 1. STATE'LERI AL VE GUARDS UYGULA
% ============================================================================

H2_aq=max(y(5),1e-12); CO2_aq=max(y(6),1e-12); SO4=max(y(7),1e-12); X=max(y(9),1e-12); S_tot=max(y(12),1e-12);
% AÇIKLAMA: İhtiyaç duyulan state'leri çıkar ve guards uygula
% DİKKAT: Tüm state'ler gerekmiyor, sadece hız hesabı için gerekenler

%%% PATCH (Fe pool): read Fe_pool
Fe_pool=max(y(14),0);
% AÇIKLAMA: Fe pool'u al (çökelme sınırlaması için)

% ============================================================================
% 2. pH VE SÜLFİD SPESİASYONU
% ============================================================================

pH=env.pH_fun(t); frac_HS=1/(1+10^(env.pKa_H2S - pH)); HS=S_tot*frac_HS;
% AÇIKLAMA: pH interpolasyonu ve HS⁻ hesabı (tek satırda)
% ADIMLAR:
%   1. pH = env.pH_fun(t): t zamanındaki pH
%   2. frac_HS = Henderson-Hasselbalch ile HS⁻ fraksiyonu
%   3. HS = S_tot * frac_HS: HS⁻ konsantrasyonu

% ============================================================================
% 3. PARAMETRELERİ AL
% ============================================================================

k_m=p(1); k_s=p(2); k_a=p(3);
% AÇIKLAMA: Hız sabitleri

KI_m=p(7); KI_s=p(8); KI_a=p(9);
% AÇIKLAMA: İnhibisyon sabitleri

k_prec=p(10); HS_sat=p(11);
% AÇIKLAMA: Çökelme parametreleri

H2_th=p(12); K_H2=p(14); K_SO4=p(15); K_CO2=p(16);
% AÇIKLAMA: Aktivasyon ve Monod sabitleri

t_lag=p(21); w_lag=p(22);
% AÇIKLAMA: Lag parametreleri

beta_SO4_m=p(24);
% AÇIKLAMA: SO4-metan rekabet parametresi

% ============================================================================
% 4. İNHİBİSYON VE AKTİVASYON
% ============================================================================

f_inh_m = KI_m/(KI_m+HS);
f_inh_s = KI_s/(KI_s+HS);
f_inh_a = KI_a/(KI_a+HS);
% AÇIKLAMA: Sülfid inhibisyon faktörleri

f_H2    = H2_aq/(H2_aq+H2_th);
% AÇIKLAMA: H2 aktivasyon faktörü

f_lag   = 1/(1+exp((t_lag - t)/max(w_lag,1e-3)));
% AÇIKLAMA: Lag sigmoid gate

f_act   = f_H2 * f_lag;
% AÇIKLAMA: Toplam aktivasyon

% ============================================================================
% 5. MONOD FAKTÖRLER
% ============================================================================

mH2  = H2_aq /(K_H2  + H2_aq);
mSO4 = SO4   /(K_SO4 + SO4);
mCO2 = CO2_aq/(K_CO2 + CO2_aq);
% AÇIKLAMA: Monod doygunluk faktörleri

% ============================================================================
% 6. REAKSİYON HIZLARI
% ============================================================================

% Match ODE definitions (include single f_comp_m on methanogenesis; SR without CO2 gate)
f_comp_m = 1 / (1 + beta_SO4_m * SO4);
% AÇIKLAMA: Methanogen-SO4 rekabet faktörü

r_meth  = k_m * X * mH2 * mCO2       * f_inh_m * f_act * f_comp_m;
# AÇIKLAMA: Methanogenesis hızı
% DİKKAT: fT_m YOK! (termodinamik gate yok, sadece kinetik)
%         Çünkü bu fonksiyon sadece plotting için, basitleştirilmiş

r_sulf  = k_s * X * mH2 * mSO4       * f_inh_s * f_act;
% AÇIKLAMA: Sulfate reduction hızı
% DİKKAT: fT_s YOK, f_comp YOK

r_aceto = k_a * X * mH2 * (mCO2.^2)  * f_inh_a * f_act;
% AÇIKLAMA: Acetogenesis hızı
% DİKKAT: fT_a YOK

% diagnostic precipitation limited by Fe_pool
r_prec_raw = k_prec * max(0, HS - HS_sat);
r_prec     = min(r_prec_raw, Fe_pool);
% AÇIKLAMA: Çökelme hızı (Fe sınırlı)

% ============================================================================
% 7. VEKTÖR DÖNDÜR
% ============================================================================

dr = [r_meth, r_sulf, r_prec, r_aceto];
% AÇIKLAMA: 4 hızı vektör olarak döndür
% BOYUT: [1x4]
% KULLANIM: Plotting ve .dat dosyası yazımı için
end

%% -------------------- Speciation helper --------------------
function [H2S_aq, HS_aq] = speciate_sulfide(S_tot, pH, pKa)
% ============================================================================
% FONKSİYON: SPECIATE_SULFIDE
% ============================================================================
% AMAÇ: Toplam sülfürü H2S ve HS⁻'e ayır
%
% INPUT:
%   S_tot : Toplam sülfür vektörü [Nx1] (mmol/L)
%   pH    : pH vektörü [Nx1] (boyutsuz)
%   pKa   : pKa değeri (scalar, örn: 7.05)
%
% OUTPUT:
%   H2S_aq : H2S konsantrasyonu [Nx1] (mmol/L)
%   HS_aq  : HS⁻ konsantrasyonu [Nx1] (mmol/L)
%
% FİZİKSEL DENGE:
%   H2S ⇌ H⁺ + HS⁻
%   Ka = [H⁺][HS⁻] / [H2S]
%   pKa = -log10(Ka)

% ============================================================================
% 1. HS⁻ FRAKSIYONUNU HESAPLA
% ============================================================================

frac_HS  = 1 ./ (1 + 10.^(pKa - pH));
% AÇIKLAMA: Henderson-Hasselbalch formülü
# FORMÜL: frac_HS = 1 / (1 + 10^(pKa - pH))
%
% ELEMENT-WİSE OPERATÖRLER (./ ve .^):
%   NEDEN: pH ve S_tot VEKTÖR (birçok zaman noktası)
%   10.^(pKa - pH): Her pH için ayrı hesap
%   1 ./ (...): Her eleman için bölme
%
% ÖRNEK:
%   pH = [7.0; 7.5; 8.0], pKa = 7.05
#   pKa - pH = [0.05; -0.45; -0.95]
%   10.^(...) = [1.12; 0.35; 0.11]
%   frac_HS = [0.47; 0.74; 0.90]
%   (pH arttıkça HS⁻ oranı artar, beklenen davranış)

% ============================================================================
% 2. HS⁻ KONSANTRASYONUabluHS_aq    = S_tot .* frac_HS;
% AÇIKLAMA: Toplam sülfürün HS⁻ kısmı
% ELEMENT-WİSE: .*
%   S_tot .* frac_HS: Her zaman noktası için çarpım
#
# ÖRNEK:
%   S_tot = [1.0; 2.0; 3.0] mmol/L
%   frac_HS = [0.47; 0.74; 0.90]
%   HS_aq = [0.47; 1.48; 2.70] mmol/L

% ============================================================================
% 3. H2S KONSANTRASYONU
% ============================================================================

H2S_aq   = S_tot - HS_aq;
% AÇIKLAMA: Geri kalan kısım H2S
% MANTIK: S_tot = H2S_aq + HS_aq
%         H2S_aq = S_tot - HS_aq
%
% ÖRNEK:
%   S_tot = [1.0; 2.0; 3.0]
%   HS_aq = [0.47; 1.48; 2.70]
%   H2S_aq = [0.53; 0.52; 0.30]
%   (pH arttıkça H2S azalır, HS⁻ artar)

end

%% -------------------- RMSE helper --------------------
function r = rmse_equal(a,b)
% ============================================================================
% FONKSİYON: RMSE_EQUAL
% ============================================================================
% AMAÇ: İki vektör/matris arasında RMSE (Root Mean Square Error) hesapla
%
% INPUT:
%   a : Model çıktısı (vektör veya matris)
%   b : Deneysel veri (aynı boyutta)
%
% OUTPUT:
#   r : RMSE değeri (scalar)
%
% FORMÜL:
%   RMSE = sqrt(mean((a - b)^2))

r = sqrt(mean((a(:)-b(:)).^2,'omitnan'));
% ============================================================================
% SATIR SATIR AÇIKLAMA
% ============================================================================
%
% 1. a(:) ve b(:):
%    AÇIKLAMA: Vektöre çevir (flatten)
%    ÖRNEK: a = [2x3 matris] -> a(:) = [6x1 vektör]
%    AMAÇ: Boyut uyumsuzluğunu önle, hepsini tek vektörde topla
%
% 2. a(:) - b(:):
%    AÇIKLAMA: Element-wise fark
%    BOYUT: [Nx1] (N = toplam eleman sayısı)
%    ÖRNEK:
#      a(:) = [1; 2; 3]
%      b(:) = [1.1; 1.9; 3.2]
%      fark = [-0.1; 0.1; -0.2]
%
% 3. (a(:) - b(:)).^2:
%    AÇIKLAMA: Her farkın karesi
%    ÖRNEK: [-0.1; 0.1; -0.2].^2 = [0.01; 0.01; 0.04]
%
% 4. mean(..., 'omitnan'):
%    AÇIKLAMA: Ortalama al
%    'omitnan': NaN değerleri göz ardı et (eksik veri varsa)
#    ÖRNEK: mean([0.01; 0.01; 0.04]) = 0.02
%
% 5. sqrt(...):
%    AÇIKLAMA: Karekök al
%    ÖRNEK: sqrt(0.02) = 0.1414
%    BİRİM: a ve b ile aynı birimde (örn: mmol)
%
% SONUÇ:
%   r = 0.1414 (ortalama hata 0.14 mmol)

% ============================================================================
# KULLANIM ÖRNEKLERİ
% ============================================================================
%
% ÖRNEK 1: Perfect fit
%   a = [1, 2, 3]
%   b = [1, 2, 3]
%   rmse_equal(a, b) = 0 (hata yok)
%
% ÖRNEK 2: Constant bias
%   a = [1, 2, 3]
%   b = [1.5, 2.5, 3.5]
%   fark = [-0.5, -0.5, -0.5]
%   rmse = sqrt(mean([0.25, 0.25, 0.25])) = sqrt(0.25) = 0.5
%
% ÖRNEK 3: NaN handling
%   a = [1, 2, NaN, 4]
%   b = [1, 2, 3, 4]
%   'omitnan' -> sadece [1,2,4] karşılaştırılır
%   rmse = 0 (NaN göz ardı edildi)

end

% ============================================================================
% HELPER FUNCTIONS ÖZETİ
% ============================================================================
%
% 1. rate_out_mixed(t, y, p, env):
%    - Belirli (t, y) için 4 reaksiyon hızını hesapla
%    - Plotting ve veri yazımı için kullanılır
%    - model_mixed'in basitleştirilmiş versiyonu (termodinamik gate'ler yok)
%
% 2. speciate_sulfide(S_tot, pH, pKa):
#    - Toplam sülfürü H2S ve HS⁻'e ayır
%    - Henderson-Hasselbalch denklemi kullanır
%    - Vektörel işlem (birçok zaman noktası için)
%
% 3. rmse_equal(a, b):
%    - RMSE metriği hesapla
%    - Model performansını ölçmek için
%    - NaN-safe (eksik veri tolere eder)

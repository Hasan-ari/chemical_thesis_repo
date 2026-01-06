% ============================================================================
% PART 5: RESIDUALS_FULL - Optimizasyon Hata Fonksiyonu
% ============================================================================
% Bu fonksiyon:
% - lsqnonlin tarafından çağrılır
% - Parametre vektörünü (p) alır
% - ODE'yi çözer, model çıktısı üretir
% - Model vs data farkını (residuals) hesaplar
% - Residual vektörünü döndürür
% ============================================================================

%% -------------------- Residuals (solve at t_exp) --------------------
function res = residuals_full(p, t_exp, data_exp, y0, env)
% ============================================================================
% FONKSİYON İMZASI
% ============================================================================
% INPUT:
%   p        : Parametre vektörü [28x1] (lsqnonlin'in denediği değerler)
%   t_exp    : Deneysel zaman noktaları [Nx1] (örn: [0, 0.5, 1, ..., 20])
%   data_exp : Deneysel veri matrisi [Nx5] (nH2_g, nCO2_g, nCH4_g, nH2S_g, SO4)
%   y0       : Başlangıç durumu [14x1]
%   env      : Çevre değişkenleri struct
#
% OUTPUT:
%   res      : Residual (hata) vektörü [N*5 x 1] (flattened)
%              AÇIKLAMA: (model - data) farkları, log transform uygulanmış
%
% AMAÇ:
%   lsqnonlin minimize eder: sum(res.^2)
%   En iyi p'yi bulmak için res'i küçültmeye çalışır

% ============================================================================
% 1. ODE ANONİM FONKSİYON OLUŞTUR
% ============================================================================

odes = @(t,y) model_mixed(t,y,p,env);
% AÇIKLAMA: ODE fonksiyonunu anonim fonksiyon olarak tanımla
% YAPISI: @(t,y) -> sadece t ve y parametre alır
%         p ve env "capture" edilir (dışarıdan)
# KULLANIM: ode15s bu fonksiyonu çağıracak
%   ode15s içinde: dydt = odes(t_current, y_current)
%   Gerçekte çağrılan: model_mixed(t_current, y_current, p, env)

% ============================================================================
% 2. ODE ÇÖZÜCÜ AYARLARI
% ============================================================================

opts = odeset('NonNegative',1:14,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
% AÇIKLAMA: ode15s için seçenekler
% odeset: MATLAB'da ODE options struct oluşturur
#
# SEÇENEKLER:
%   'NonNegative', 1:14
%     AÇIKLAMA: State'lerin 1-14'ü negatif olamaz
%     NEDEN: Mol miktarları ve konsantrasyonlar negatif olamaz
%     ETKİ: Eğer bir state negatif olmaya çalışırsa, solver onu 0'a çeker
%
%   'RelTol', 1e-8
%     AÇIKLAMA: Relative tolerance (bağıl hata toleransı)
%     ANLAMLI: %0.000001 hata kabul edilebilir
%     KÜÇÜK DEĞER: Daha hassas çözüm (ama yavaş)
#
%   'AbsTol', 1e-10
%     AÇIKLAMA: Absolute tolerance (mutlak hata toleransı)
%     ANLAMLI: 0.0000000001 mutlak hata
%     KÜÇÜK DEĞER: Çok küçük sayılarda bile hassasiyet
%
%   'MaxStep', 0.5
%     AÇIKLAMA: Maksimum zaman adımı (gün)
%     NEDEN: Çok büyük adım -> hızlı değişimleri kaçırabilir
%     0.5 GÜN: Yarım günlük adımdan büyük atılmaz

% ============================================================================
% 3. ODE ÇÖZÜMÜ
% ============================================================================

[~, y_sim] = ode15s(odes, t_exp, y0, opts);
% AÇIKLAMA: ODE sistemini çöz
# ode15s: MATLAB'ın stiff ODE solver'ı (katı sistemler için)
%
% PARAMETRELER:
%   odes    : @(t,y) fonksiyon handle
%   t_exp   : Çözümün istenen zaman noktaları [0, 0.5, 1, ..., 20]
%   y0      : Başlangıç durumu [14x1]
#   opts    : Çözücü seçenekleri
%
% ÇIKTILAR:
%   [~, y_sim]:
%     ~: Zaman vektörü (kullanmıyoruz, çünkü t_exp ile aynı)
%     y_sim: Çözüm matrisi [N x 14]
#            Her satır bir zaman noktası
%            Her sütun bir state variable
#
# ÖRNEK:
#   t_exp = [0; 1; 2; 3] (4 zaman noktası)
%   y_sim = [14 sütunlu, 4 satırlı matris]
%   y_sim(1,:) = t=0'daki 14 state
%   y_sim(2,:) = t=1'deki 14 state
%   ...

% ============================================================================
% 4. MODEL ÇIKTISINI EKSTRAKSİYON
# ============================================================================

% Compare model to: nH2_g, nCO2_g, nCH4_g, nH2S_g, SO4
sim_mat = [y_sim(:,1), y_sim(:,2), y_sim(:,3), y_sim(:,4), y_sim(:,7)];
% AÇIKLAMA: y_sim'den sadece ilgilendiğimiz 5 sütunu al
% SÜTUNLAR:
%   y_sim(:,1) = nH2_g   (tüm satırlar, 1. sütun)
%   y_sim(:,2) = nCO2_g
%   y_sim(:,3) = nCH4_g
%   y_sim(:,4) = nH2S_g
%   y_sim(:,7) = SO4
# DİKKAT: 5,6,8,9,... sütunları dahil değil (onları fit etmiyoruz)
%
% BOYUT: [N x 5]
%   N: Zaman noktası sayısı (t_exp'in uzunluğu)
%   5: Karşılaştırılan değişken sayısı

% ============================================================================
% 5. LOG TRANSFORM - Model ve Data
% ============================================================================

log_sim = log1p(sim_mat);
% AÇIKLAMA: Model çıktısına log1p transform uygula
# log1p(x) = log(1 + x)
%   NEDEN log1p: log(0) = -Inf (sorun!), ama log1p(0) = 0 (güvenli)
%   AMAÇ: Büyük ve küçük değerleri dengele
%   ÖRNEK:
%     sim_mat = [0.01, 100, 1000]
%     log1p   = [0.01, 4.62, 6.91] (farklar daha dengeli)
%
% ELEMENT-WİSE: Her elemana ayrı ayrı uygulanır
% BOYUT: [N x 5] (sim_mat ile aynı)

log_exp = log1p([data_exp(:,1), data_exp(:,2), data_exp(:,3), data_exp(:,4), data_exp(:,5)]);
% AÇIKLAMA: Deneysel veriye aynı transform
% DİKKAT: data_exp zaten [N x 5] formatında
%         (nH2_g, nCO2_g, nCH4_g, nH2S_g, SO4)

% ============================================================================
% 6. AĞIRLIKLANDIRMA
% ============================================================================

% %%% PATCH earlier: emphasize SO4 plateau more strongly (was 1.1)
weights = [1, 1, 0.9, 1.0, 2.0];
# AÇIKLAMA: Her değişken için ağırlık faktörü
% VEKTÖR: [wH2, wCO2, wCH4, wH2S, wSO4]
%   1.0: H2'ye normal ağırlık
%   1.0: CO2'ye normal ağırlık
%   0.9: CH4'e biraz daha az ağırlık (daha toleranslı)
%   1.0: H2S'e normal ağırlık
%   2.0: SO4'e çift ağırlık (daha önemli!)
%
% AMAÇ: SO4'ün daha iyi fit olmasını sağla
%   Çünkü SO4 plateau'su kritik
%
% KULLANIM: res = (log_sim - log_exp) .* weights
%   Element-wise çarpım: Her sütun kendi ağırlığı ile çarpılır

% ============================================================================
% 7. RESIDUAL (HATA) HESABI
% ============================================================================

res = (log_sim - log_exp) .* weights;
% AÇIKLAMA: Ağırlıklı hata matrisi
# FORMÜL: res(i,j) = (log_sim(i,j) - log_exp(i,j)) * weights(j)
%   i: Zaman indeksi (1'den N'e)
%   j: Değişken indeksi (1'den 5'e)
%
% ELEMENT-WİSE OPERATÖRLER:
%   (log_sim - log_exp): Matrix subtraction [N x 5]
%   .* weights: Element-wise multiplication
%     MATLAB broadcasting: weights [1x5] otomatik tekrarlanır [Nx5]'e
%
% BOYUT: res = [N x 5]
#
% ÖRNEK:
%   N = 40 zaman noktası
%   res = [40 x 5] matris
%   res(10, 3) = t=10'daki CH4 hatası * 0.9

% ============================================================================
% 8. NEGATİF STATE CEZASI
% ============================================================================

% Elimination for negative states
if any(y_sim(:) < -1e-9), res = res + 1e3 * abs(min(y_sim(:))); end
% AÇIKLAMA: Eğer herhangi bir state negatif ise, ceza ekle
#
# ADIM ADIM:
%   1. y_sim(:): Tüm matrisi vektöre çevir (column-major order)
%      BOYUT: [N*14 x 1] (tüm state'ler tek vektörde)
%
%   2. y_sim(:) < -1e-9: Hangisi çok negatif? (boolean vektör)
%      EŞIK: -1e-9 = -0.000000001 (numerik hata toleransı)
%      NEDEN -1e-9: Çok küçük negatifler (roundoff error) kabul edilir
#
#   3. any(...): Hiç TRUE var mı? (boolean scalar)
%      TRUE ise: En az bir state çok negatif
%
%   4. if TRUE:
%      min(y_sim(:)): En küçük değer (en negatif olan)
%      abs(min(...)): Mutlak değer (pozitif yap)
#      1e3 * ...: 1000 ile çarp (büyük ceza)
#      res = res + ceza: Tüm res matrisine ceza ekle
#
% AMAÇ: Optimizer'ı negatif state'lerden uzak tut
#   Eğer p değerleri kötüyse -> ODE patlayabilir -> negatif state
%   -> Büyük ceza -> Optimizer bu p'yi kullanmaz
#
# ÖRNEK:
%   min(y_sim(:)) = -0.05 (çok negatif, SO4 belki)
%   abs(-0.05) = 0.05
%   1e3 * 0.05 = 50 (büyük ceza!)
%   res'in her elemanına +50 eklenir
%   sum(res.^2) çok büyür -> bu p reddedilir

% ============================================================================
# 9. FLATTEN (VEKTÖRE ÇEVİR)
% ============================================================================

res = res(:);
% AÇIKLAMA: Matrisi tek sütun vektöre çevir
# İŞLEM: [N x 5] -> [N*5 x 1]
%   ÖRNEK: N=40, 5 değişken
%          [40 x 5] -> [200 x 1]
#
# SIRA (column-major):
%   res(1:40)    = 1. değişken (H2) tüm zamanlar
%   res(41:80)   = 2. değişken (CO2) tüm zamanlar
#   res(81:120)  = 3. değişken (CH4) tüm zamanlar
#   res(121:160) = 4. değişken (H2S) tüm zamanlar
%   res(161:200) = 5. değişken (SO4) tüm zamanlar
#
% NEDEN:
#   lsqnonlin VEKTÖR bekler (matris değil)
%   sum(res.^2) tek bir skalar sayı olmalı

% ============================================================================
% FONKSİYON BİTİŞİ
% ============================================================================
end
% AÇIKLAMA: residuals_full fonksiyonu burada biter
#
# ÖZET AKIŞ:
#   1. p parametresiyle ODE'yi çöz
#   2. y_sim çözümünü al
#   3. İlgili 5 sütunu çıkar (sim_mat)
#   4. log1p transform uygula (log_sim, log_exp)
#   5. Farkı hesapla ve ağırlıklandır
#   6. Negatif state varsa ceza ekle
#   7. Vektöre çevir ve döndür
#
# lsqnonlin KULLANIMI:
#   Her iterasyonda:
#     res = residuals_full(p_current, ...)
#     error = sum(res.^2)
#     Gradient hesapla (Jacobian)
#     p_next = p_current - step * gradient
#   Hedef: error'u minimize et

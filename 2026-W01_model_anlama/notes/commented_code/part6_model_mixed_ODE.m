% ============================================================================
% PART 6: MODEL_MIXED - ANA ODE FONKSİYONU
% ============================================================================
% Bu fonksiyon:
% - Sistemin KALP ATIŞI (core logic)
% - ode15s solver her zaman adımında bu fonksiyonu çağırır
% - Mevcut durumu (y) alır, türevleri (dydt) hesaplar
% - 14 diferansiyel denklem çözer
% ============================================================================

%% -------------------- ODE model (with Fe pool & H2S outgassing-positive flux) --------------------
function dydt = model_mixed(t, y, p, env)
% ============================================================================
% FONKSİYON İMZASI (Function Signature)
% ============================================================================
% AÇIKLAMA: Bu, ODE çözücüsünün çağıracağı fonksiyon
% MATLAB ODE FONKSİYON FORMATI: dydt = f(t, y)
%
% INPUT:
%   t    : Zaman (scalar, gün cinsinden)
%          ÖRNEK: t = 2.5 (2.5 gün zamanındayız)
%   y    : Durum vektörü [14x1] (veya [1x14] - MATLAB otomatik çevirir)
%          ÖRNEK: y = [nH2_g, nCO2_g, ..., Fe_pool]
%   p    : Parametre vektörü [28x1]
%          ÖRNEK: p = p_fit veya p0 (optimize edilmiş veya başlangıç)
%   env  : Çevre değişkenleri struct
%          ÖRNEK: env.Vg, env.T, env.pH_fun(t)
%
% OUTPUT:
%   dydt : Türev vektörü [14x1]
%          AÇIKLAMA: Her state'in zaman türevi
%          ÖRNEK: dydt(1) = dnH2_g/dt (H2 gazının değişim hızı)
%
% ODE15S ÇAĞRISI:
%   Her zaman adımında (örn: t=0, 0.01, 0.02, ... 20 gün)
%   solver: dydt = model_mixed(t, y_current, p_fit, env) çağırır
%   sonra: y_next = y_current + dydt * dt (yaklaşık olarak)

% ============================================================================
% 1. UNPACK - env Struct'tan Değişkenleri Çıkar
% ============================================================================

Vg=env.Vg; Vl=env.Vl; T=env.T; Rgas=env.Rgas;
% AÇIKLAMA: Struct alanlarını lokal değişkenlere ata
% NEDEN: Her seferinde "env.Vg" yazmak yerine "Vg" yazmak daha kolay
% İŞLEYİŞ:
%   env.Vg     (struct access) -> Vg      (local variable)
%   env.Vl     (struct access) -> Vl      (local variable)
%   env.T      (struct access) -> T       (local variable)
%   env.Rgas   (struct access) -> Rgas    (local variable)

Hcp_H2=env.Hcp_H2_eff; Hcp_CO2=env.Hcp_CO2_eff; Hcp_H2S=env.Hcp_H2S_eff;
% AÇIKLAMA: Henry sabitlerini unpack et
% NOT: Efektif Henry sabitleri (phi ile çarpılmış hali)

pH=env.pH_fun(t); pKa=env.pKa_H2S;
% AÇIKLAMA: pH ve pKa değerlerini al
% ÖNEMLİ: pH_fun bir FONKSİYON, çağırmak için (t) gerekli
%   env.pH_fun(t): t zamanındaki pH değerini döndürür (interpolasyon)
%   ÖRNEK: t=2.5 ise -> pH = 7.32 (interpolasyon sonucu)
% pKa sabit bir sayı (7.05)

% ============================================================================
% 2. STATE VEKTÖRÜNÜ UNPACK ET
% ============================================================================
% AÇIKLAMA: y vektöründen her bir state'i çıkar
% y = [14 elemanlı vektör]
%   y(1), y(2), ..., y(14)

% State
nH2_g=y(1); nCO2_g=y(2); nCH4_g=y(3); nH2S_g=y(4);
% AÇIKLAMA: Gaz fazındaki mol miktarları (mmol)
%   nH2_g   = y(1) : Headspace'deki H2 (mmol)
%   nCO2_g  = y(2) : Headspace'deki CO2 (mmol)
%   nCH4_g  = y(3) : Headspace'deki CH4 (mmol)
%   nH2S_g  = y(4) : Headspace'deki H2S (mmol)

H2_aq=y(5); CO2_aq=y(6); SO4=y(7); FeS=y(8); X=y(9);
% AÇIKLAMA: Sıvı fazdaki konsantrasyonlar (mmol/L)
%   H2_aq   = y(5) : Çözünmüş H2 (mmol/L)
%   CO2_aq  = y(6) : Çözünmüş CO2 (mmol/L)
%   SO4     = y(7) : Sülfat (mmol/L)
%   FeS     = y(8) : Çökelen demir sülfür (mmol/L)
%   X       = y(9) : Biomass (mmol/L)

Ac=y(10); HCO3=y(11); S_tot=y(12); Lag=y(13);
% AÇIKLAMA: Diğer state'ler
%   Ac      = y(10) : Acetate (mmol/L)
%   HCO3    = y(11) : Bicarbonate (mmol/L)
%   S_tot   = y(12) : Toplam çözünmüş sülfür H2S(aq)+HS⁻ (mmol/L)
%   Lag     = y(13) : Lag variable (0-1 arası, aktivasyon tracking)

%%% PATCH (Fe pool): add Fe_pool state
Fe_pool = y(14);
% AÇIKLAMA: Çözünmüş demir havuzu (mmol/L)
%   Fe_pool = y(14) : Fe²⁺ konsantrasyonu

% ============================================================================
% 3. PARAMETRE VEKTÖRÜNÜ UNPACK ET
% ============================================================================
% AÇIKLAMA: p vektöründen her bir parametreyi çıkar
% p = [28 elemanlı vektör]

% Parameters
k_m  = p(1);  k_s  = p(2);  k_a  = p(3);
% AÇIKLAMA: Maksimum hız sabitleri (1/gün)
%   k_m  = p(1) : Methanogenesis hız sabiti
%   k_s  = p(2) : Sulfate reduction hız sabiti
%   k_a  = p(3) : Acetogenesis hız sabiti

Y_m  = p(4);  Y_s  = p(5);  Y_a  = p(6);
% AÇIKLAMA: Yield katsayıları (biomass/substrate)
%   Y_m  = p(4) : Methanogenesis için yield
%   Y_s  = p(5) : Sulfate reduction için yield
%   Y_a  = p(6) : Acetogenesis için yield

KI_m = p(7);  KI_s = p(8);  KI_a = p(9);
% AÇIKLAMA: İnhibisyon sabitleri (mmol/L)
%   KI_m = p(7) : Methanogenesis için HS⁻ inhibisyon sabiti
%   KI_s = p(8) : Sulfate reduction için HS⁻ inhibisyon sabiti
%   KI_a = p(9) : Acetogenesis için HS⁻ inhibisyon sabiti

k_prec = p(10); % FeS precipitation kinetic factor
% AÇIKLAMA: FeS çökelme hızı (1/gün)
HS_sat = p(11); H2_th = p(12); DG_th = p(13);
% AÇIKLAMA: Eşik değerler
%   HS_sat  = p(11) : HS⁻ doygunluk eşiği (çökelme için) (mmol/L)
%   H2_th   = p(12) : H2 aktivasyon eşiği (mmol/L)
%   DG_th   = p(13) : Termodinamik eşik (kJ/mol)

K_H2 = p(14); K_SO4 = p(15); K_CO2 = p(16);
% AÇIKLAMA: Monod yarı-doygunluk sabitleri (mmol/L)
%   K_H2   = p(14) : H2 için Monod sabiti
%   K_SO4  = p(15) : SO4 için Monod sabiti
%   K_CO2  = p(16) : CO2 için Monod sabiti

kla_H2 = p(17); kla_CO2 = p(18); kla_H2S = p(19);
% AÇIKLAMA: Gaz-sıvı kütle transfer katsayıları (1/gün)
%   kla_H2  = p(17) : H2 transfer katsayısı
%   kla_CO2 = p(18) : CO2 transfer katsayısı
%   kla_H2S = p(19) : H2S transfer katsayısı

b = p(20); t_lag = p(21); w_lag = p(22);
% AÇIKLAMA: Biomass ve lag parametreleri
%   b     = p(20) : Biomass decay katsayısı (1/gün)
%   t_lag = p(21) : Lag merkez zamanı (gün)
%   w_lag = p(22) : Lag genişliği (gün)

% %%% NEW () indices:
k_diss_gyp = p(23); % 1/day
% AÇIKLAMA: Gypsum çözünme hızı (1/gün)

beta_SO4_m = p(24); % mM^-1
% AÇIKLAMA: SO4-Methanogen rekabet katsayısı (1/mM)

% AÇIKLAMA: Henry düzeltme faktörleri ve H2S çarpanı
% (Kodda kullanılmıyor çünkü zaten env'de Hcp_eff var, ama fitting için gerekli)
% phi_H2 = p(25); phi_CO2 = p(26); phi_H2S = p(27); alpha_H2S = p(28);

% ============================================================================
% 4. TERMODİNAMİK SABİTLER
% ============================================================================

% Thermodynamics
RkJ=8.314e-3; RT=RkJ*T;
% AÇIKLAMA: Termodinamik hesaplar için sabitler
%   RkJ  = 8.314e-3 : Gaz sabiti R (kJ/(mol·K))
%          (NOT: 8.314 J/(mol·K) = 0.008314 kJ/(mol·K))
%   RT   = R * T : Enerji birimi (kJ/mol)
%          ÖRNEK: 0.008314 * 298.15 = 2.479 kJ/mol
% KULLANIM: Gibbs free energy hesaplarında DG = DG0 + RT*ln(Q)

DG0_m=-130; DG0_s=-152; DG0_a=-95;
% AÇIKLAMA: Standart Gibbs serbest enerji değişimi (kJ/mol)
%   DG0_m = -130 : Methanogenesis için DG° (negatif = ekzotermik)
%   DG0_s = -152 : Sulfate reduction için DG°
%   DG0_a = -95  : Acetogenesis için DG°
% FİZİKSEL ANLAM: Negatif = enerji salınır (thermodynamically favorable)

% ============================================================================
% 5. GUARDS - Numerik Stabilite İçin Korumalar
% ============================================================================

% Guards
eps=1e-12;
% AÇIKLAMA: Epsilon = çok küçük pozitif sayı (0.000000000001)
% AMAÇ: Bölme hatalarını önlemek (0'a bölme yasak!)

nH2_g=max(nH2_g,eps); nCO2_g=max(nCO2_g,eps); nCH4_g=max(nCH4_g,eps); nH2S_g=max(nH2S_g,eps);
% AÇIKLAMA: Gaz miktarlarını eps'den küçük olma
% FORMÜL: max(x, eps) -> eğer x < eps ise eps döndür, değilse x döndür
% NEDEN: İdeal gaz denkleminde P = nRT/V var, eğer n=0 ise P=0 ama
%        log(P) veya 1/P hesabında sorun çıkar
% ÖRNEK:
%   nH2_g = -0.001 (negatif, ODE hatası) -> max(-0.001, 1e-12) = 1e-12 (düzelt)
%   nH2_g = 5.0 (pozitif) -> max(5.0, 1e-12) = 5.0 (değişmez)

H2_aq=max(H2_aq,eps); CO2_aq=max(CO2_aq,eps); SO4=max(SO4,eps); S_tot=max(S_tot,eps);
% AÇIKLAMA: Konsantrasyonlar için aynı koruma

Ac=max(Ac,eps); HCO3=max(HCO3,eps); X=max(X,eps); Fe_pool=max(Fe_pool,0);
% AÇIKLAMA: Diğer state'ler için koruma
% DİKKAT: Fe_pool için max(..., 0) kullanılmış (eps yerine)
%   NEDEN: Fe_pool 0 olabilir (tükenebilir), negatif olmamalı

% ============================================================================
% 6. KISMİ BASINÇLAR - İdeal Gaz Denklemi
% ============================================================================

% Partial pressures (atm) from moles (mmol)
pH2  = (nH2_g /1000)  * Rgas * T / Vg;
% AÇIKLAMA: H2 kısmi basıncını hesapla
% FORMÜL: P = nRT/V
%   n: mol sayısı (nH2_g/1000 ile mmol->mol dönüşümü)
%   R: Gaz sabiti (0.082057 L·atm/(mol·K))
%   T: Sıcaklık (298.15 K)
%   V: Hacim (Vg = 0.14 L)
% ADIM ADIM:
%   nH2_g = 5 mmol olsun
%   nH2_g/1000 = 0.005 mol
%   0.005 * 0.082057 * 298.15 = 0.1224 atm·L
%   0.1224 / 0.14 = 0.874 atm

pCO2 = (nCO2_g/1000)  * Rgas * T / Vg;
pH2S = (nH2S_g/1000)  * Rgas * T / Vg;
% AÇIKLAMA: Aynı formül CO2 ve H2S için

% ============================================================================
% 7. HENRY DENGESİ - Gaz-Sıvı Dengesi
% ============================================================================

% Henry equilibria (mmol/L) @ 25 °C
Ceq_H2  = Hcp_H2  * pH2;
% AÇIKLAMA: Denge halinde sıvıdaki H2 konsantrasyonu
% FORMÜL: C_eq = H_cp * P (Henry Yasası)
%   H_cp: Henry sabiti (mmol/L/atm)
%   P: Gaz basıncı (atm)
% ÖRNEK:
%   pH2 = 0.874 atm
%   Hcp_H2 = 0.78 mmol/L/atm
%   Ceq_H2 = 0.78 * 0.874 = 0.682 mmol/L
% FİZİKSEL ANLAM: Bu basınçta, denge halindeki sıvıda 0.682 mmol/L H2 olur

Ceq_CO2 = Hcp_CO2 * pCO2;
Ceq_H2S = Hcp_H2S * pH2S;
% AÇIKLAMA: Aynı mantık CO2 ve H2S için

% ============================================================================
% 8. GAZ-SIVI KÜTLE TRANSFERİ
% ============================================================================

% Gas–liquid transfers for H2, CO2 (liquid-side uptake positive)
J_H2  = kla_H2  * (Ceq_H2  - H2_aq);
% AÇIKLAMA: H2 için kütle transfer hızı (mmol/L/gün)
% FORMÜL: J = kla * (C_eq - C_aq)
%   kla: Kütle transfer katsayısı (1/gün)
%   C_eq: Denge konsantrasyonu (Henry'den)
%   C_aq: Gerçek sıvı konsantrasyonu
% MANTIK:
%   C_eq > C_aq ise: Gazdan sıvıya transfer (J > 0, uptake)
%   C_eq < C_aq ise: Sıvıdan gaza transfer (J < 0, degassing)
% ÖRNEK:
%   Ceq_H2 = 0.682 mmol/L
%   H2_aq = 0.5 mmol/L
%   kla_H2 = 10 1/day
%   J_H2 = 10 * (0.682 - 0.5) = 1.82 mmol/L/day (sıvıya giriyor)

J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq);
% AÇIKLAMA: Aynı formül CO2 için

% ============================================================================
% 9. SÜLFİD SPESİASYONU - H2S ⇌ H⁺ + HS⁻
% ============================================================================

% Sulfide speciation
frac_HS  = 1/(1+10^(pKa - pH));
% AÇIKLAMA: HS⁻ (deprotonize form) fraksiyonunu hesapla
% FORMÜL: Henderson-Hasselbalch denkleminden türetilmiş
%   frac_HS = 1 / (1 + 10^(pKa - pH))
% DENGE: H2S ⇌ H⁺ + HS⁻
%   pKa = 7.05: Bu pH'da yarı-yarıya dağılır
%   pH > pKa ise: HS⁻ dominant (frac_HS > 0.5)
%   pH < pKa ise: H2S dominant (frac_HS < 0.5)
% ÖRNEK:
%   pH = 7.5, pKa = 7.05
%   pKa - pH = 7.05 - 7.5 = -0.45
%   10^(-0.45) = 0.355
%   frac_HS = 1 / (1 + 0.355) = 0.738 (73.8% HS⁻ formunda)

frac_H2S = 1 - frac_HS;
% AÇIKLAMA: H2S (protonize form) fraksiyonu
% MANTIK: Toplam = 1, yani frac_H2S + frac_HS = 1

HS_aq  = S_tot*frac_HS;
% AÇIKLAMA: HS⁻ konsantrasyonu (mmol/L)
% FORMÜL: HS⁻ = S_tot * frac_HS
% ÖRNEK:
%   S_tot = 2.0 mmol/L
#   frac_HS = 0.738
%   HS_aq = 2.0 * 0.738 = 1.476 mmol/L

H2S_aq = S_tot*frac_H2S;
% AÇIKLAMA: H2S konsantrasyonu (mmol/L)

% ============================================================================
% 10. H2S GAZLAŞMA (DEGASSING)
% ============================================================================

% H2S: outgassing-positive flux
Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S);
% AÇIKLAMA: H2S için kütle transfer hızı
% DİKKAT: J_H2 ve J_CO2'den FARKLI!
%   - J_H2, J_CO2: Ceq - C_aq (denge - gerçek)
%   - Jout_H2S: C_aq - Ceq (gerçek - denge)
% NEDEN FARK:
%   - H2S genelde sıvıdan gaza gider (degassing)
%   - Pozitif Jout_H2S = sıvıdan gazagidiyor
% FORMÜL: Jout_H2S = kla * (H2S_aq - C_eq)
%   H2S_aq > Ceq ise: Jout > 0 (sıvıdan gaza, degassing)
%   H2S_aq < Ceq ise: Jout < 0 (gazdan sıvıya)

% ============================================================================
% 11. İNHİBİSYON VE AKTİVASYON FONKSIYONLARI
% ============================================================================

% Inhibitions & Activation: dissolved H2 + smooth lag gate
f_inh_m = KI_m/(KI_m+HS_aq);
% AÇIKLAMA: Methanogenesis için sülfid inhibisyon faktörü
% FORMÜL: f_inh = KI / (KI + HS)
%   KI: İnhibisyon sabiti (mmol/L)
%   HS: HS⁻ konsantrasyonu (mmol/L)
% DAVRANIŞI:
%   HS_aq = 0 ise: f_inh = KI/KI = 1 (inhibisyon yok, tam hız)
%   HS_aq >> KI ise: f_inh ≈ 0 (güçlü inhibisyon, hız sıfıra yakın)
%   HS_aq = KI ise: f_inh = 0.5 (yarı hız)
% TİP: Yarışmalı (competitive) inhibisyon benzeri

f_inh_s = KI_s/(KI_s+HS_aq);
f_inh_a = KI_a/(KI_a+HS_aq);
% AÇIKLAMA: Sulfate reduction ve acetogenesis için aynı mantık

f_H2    = H2_aq/(H2_aq+H2_th);
% AÇIKLAMA: H2 aktivasyon faktörü
% FORMÜL: f_H2 = H2 / (H2 + H2_th)
%   H2_th: Eşik H2 konsantrasyonu
% DAVRANIŞI:
%   H2_aq = 0 ise: f_H2 = 0 (H2 yok, aktivasyon yok)
%   H2_aq >> H2_th ise: f_H2 ≈ 1 (bol H2, tam aktivasyon)
%   H2_aq = H2_th ise: f_H2 = 0.5 (yarı aktivasyon)
% TİP: Hill tipi (Monod benzeri) aktivasyon

f_lag   = 1/(1+exp((t_lag - t)/max(w_lag,1e-3)));
% AÇIKLAMA: Lag fazı geçiş fonksiyonu (sigmoid)
% FORMÜL: f_lag = 1 / (1 + exp((t_lag - t) / w_lag))
%   t_lag: Merkez zaman (gün) - geçişin orta noktası
#   w_lag: Genişlik (gün) - geçişin ne kadar hızlı olduğu
%   t: Mevcut zaman
% DAVRANIŞI:
%   t << t_lag ise: exp((t_lag-t)/w_lag) çok büyük -> f_lag ≈ 0 (lag fazı)
%   t >> t_lag ise: exp((t_lag-t)/w_lag) çok küçük -> f_lag ≈ 1 (aktif faz)
%   t = t_lag ise: exp(0) = 1 -> f_lag = 0.5 (yarı geçiş)
% GRAFİK: S-şekli (sigmoid curve)
% max(w_lag, 1e-3): w_lag çok küçükse 1e-3 kullan (bölme hatasını önle)

f_act   = f_H2 * f_lag;
% AÇIKLAMA: Toplam aktivasyon faktörü
% FORMÜL: f_act = f_H2 × f_lag
%   Her ikisi de 0-1 arası olduğu için f_act de 0-1 arası
% MANTIK: Hem H2 olmalı hem de lag fazı geçmiş olmalı

% ============================================================================
% 12. MONOD DOYGUNLUK FONKSİYONLARI
% ============================================================================

% Monod saturations
mH2  = H2_aq /(K_H2  + H2_aq);
% AÇIKLAMA: H2 için Monod doygunluğu
% FORMÜL: m = S / (K + S)
%   S: Substrat konsantrasyonu (H2_aq)
%   K: Yarı-doygunluk sabiti (K_H2)
% DAVRANIŞI:
%   H2_aq = 0 ise: m = 0 (substrat yok)
%   H2_aq >> K ise: m ≈ 1 (doymuş, maksimum hız)
%   H2_aq = K ise: m = 0.5 (yarı maksimum hız)
% TİP: Michaelis-Menten kinetics (enzyme kinetics)

mSO4 = SO4   /(K_SO4 + SO4);
mCO2 = CO2_aq/(K_CO2 + CO2_aq);
% AÇIKLAMA: SO4 ve CO2 için aynı mantık

% ============================================================================
% 13. TERMODİNAMİK GATE FONKSİYONLARI
% ============================================================================

% Thermo gates (heuristics)
%  SR should not depend on CO2/HCO3
Q_s = 1;
% AÇIKLAMA: Sulfate reduction için reaksiyon bölümü (quotient)
% SABİT = 1: SR için termodinamik kontrol yok (her zaman favorable)

Q_a = Ac / (H2_aq^4 * CO2_aq^2);
% AÇIKLAMA: Acetogenesis için reaksiyon bölümü
% FORMÜL: Q = [Ürün] / [Reaktant]^stoich
%   Ac: Acetate konsantrasyonu (ürün)
%   H2_aq^4: H2 konsantrasyonunun 4. kuvveti (reaktant)
%   CO2_aq^2: CO2 konsantrasyonunun 2. kuvveti (reaktant)
# REAKSIYON: 4H2 + 2CO2 → Acetate + ...
% YÜKSEK Q: Ürün bol, reaktant az -> reaksiyon geri gitmek ister (unfavorable)
% DÜŞÜK Q: Ürün az, reaktant bol -> reaksiyon ileri gider (favorable)

DG_s = DG0_s + RT*log(Q_s);
% AÇIKLAMA: Sulfate reduction için Gibbs serbest enerji
% FORMÜL: ΔG = ΔG° + RT ln(Q)
%   ΔG°: Standart Gibbs enerji (-152 kJ/mol)
%   RT: 2.479 kJ/mol
%   ln(Q): Q'nun doğal logaritması (Q=1 ise ln(1)=0)
% SONUÇ: DG_s = -152 + 0 = -152 kJ/mol (sabit, Q=1 olduğu için)

DG_m = DG0_m;
% AÇIKLAMA: Methanogenesis için ΔG
% NOT: Q hesaplanmamış, direkt DG0 kullanılmış
# SONUÇ: DG_m = -130 kJ/mol (sabit)

DG_a = DG0_a + RT*log(Q_a);
% AÇIKLAMA: Acetogenesis için ΔG
% FORMÜL: ΔG = -95 + 2.479*ln(Q_a)
% DAVRANIŞI:
%   Q_a küçük ise: ln(Q_a) negatif -> DG daha negatif (favorable)
%   Q_a büyük ise: ln(Q_a) pozitif -> DG daha az negatif (unfavorable)

fT_s = 1/(1+exp((DG_s - DG_th)/RT));
% AÇIKLAMA: Sulfate reduction için termodinamik gate
% FORMÜL: fT = 1 / (1 + exp((ΔG - ΔG_th) / RT))
%   ΔG: Gerçek Gibbs enerji
%   ΔG_th: Eşik enerji (örn: -12 kJ/mol)
%   RT: 2.479 kJ/mol
% DAVRANIŞI (Sigmoid):
#   ΔG << ΔG_th ise: exp(...) çok küçük -> fT ≈ 1 (thermodynamically favorable)
%   ΔG >> ΔG_th ise: exp(...) çok büyük -> fT ≈ 0 (unfavorable, reaksiyon durur)
%   ΔG = ΔG_th ise: exp(0) = 1 -> fT = 0.5
% ÖRNEK:
%   DG_s = -152 kJ/mol, DG_th = -12 kJ/mol
%   (-152 - (-12)) / 2.479 = -140 / 2.479 = -56.5
%   exp(-56.5) ≈ 0 (çok küçük)
%   fT_s ≈ 1 / (1 + 0) = 1 (tam hızda gidebilir)

fT_m = 1/(1+exp((DG_m - DG_th)/RT));
fT_a = 1/(1+exp((DG_a - DG_th)/RT));
% AÇIKLAMA: Methanogenesis ve acetogenesis için aynı gate

% ============================================================================
% 14. SÜLFAT-METAN REKABETİ
% ============================================================================

% Sulfate vs methanogen competition gate
f_comp_m = 1 / (1 + beta_SO4_m * SO4);
% AÇIKLAMA: Methanogenesis için SO4 baskılama faktörü
% FORMÜL: f_comp = 1 / (1 + β * [SO4])
#   β (beta_SO4_m): Rekabet katsayısı (1/mM)
%   SO4: Sülfat konsantrasyonu (mM)
% DAVRANIŞI:
%   SO4 = 0 ise: f_comp = 1 (rekabet yok, methanogen tam hızda)
%   SO4 yüksek ise: f_comp ≈ 0 (sülfat var, methanogen baskılanır)
% BİYOLOJİK ANLAM: Sülfat indirgeyiciler daha avantajlı, methanogenler rekabet edemez

% ============================================================================
% 15. REAKSİYON HIZLARI
% ============================================================================

% Biomass-mediated rates (mmol/L/day)
r_meth  = k_m * X * mH2 * mCO2       * f_inh_m * f_act * fT_m * f_comp_m;
% AÇIKLAMA: Methanogenesis hızı (mmol/L/gün)
% FORMÜL: r = k * X * [Monod] * [Inhibitions] * [Activations] * [Thermodynamics] * [Competition]
% BILEŞENLER:
%   k_m: Maksimum hız sabiti (1/gün)
%   X: Biomass konsantrasyonu (mmol/L)
%   mH2: H2 Monod faktörü (0-1)
%   mCO2: CO2 Monod faktörü (0-1)
%   f_inh_m: HS⁻ inhibisyonu (0-1)
%   f_act: Aktivasyon (H2 + lag) (0-1)
%   fT_m: Termodinamik gate (0-1)
%   f_comp_m: SO4 rekabet gate (0-1)
% TÜM FAKTÖRLER ÇARPILIR: Hepsi 0-1 arası olduğu için:
%   - Hepsi 1 ise: r_meth = k_m * X (maksimum hız)
%   - Herhangi biri 0 ise: r_meth = 0 (reaksiyon durur)
%   - Genelde: r_meth << k_m * X (birçok sınırlama var)

r_sulf  = k_s * X * mH2 * mSO4       * f_inh_s * f_act * fT_s;          % SR without CO2 gating
% AÇIKLAMA: Sulfate reduction hızı
% DİKKAT: mCO2 YOK! (SR CO2'ye bağlı değil)
% DİKKAT: f_comp YOK! (SR kendini baskılamaz)

r_aceto = k_a * X * mH2 * (mCO2.^2)  * f_inh_a * f_act * fT_a;
% AÇIKLAMA: Acetogenesis hızı
# DİKKAT: mCO2.^2 (CO2'nin karesi!) - reaksiyon 2CO2 gerektiriyor

% ============================================================================
% 16. FeS ÇÖKELMESI
% ============================================================================

% Precipitation (from HS-), limited by Fe pool (1:1 Fe:HS stoichiometry)
r_prec_raw = k_prec * max(0, HS_aq - HS_sat);
% AÇIKLAMA: Ham çökelme hızı (mmol/L/gün)
% FORMÜL: r_prec = k_prec * max(0, HS - HS_sat)
%   k_prec: Çökelme hız katsayısı (1/gün)
%   HS_aq - HS_sat: Doygunluk aşımı (mmol/L)
%   max(0, ...): Negatif olamaz (sadece HS > HS_sat ise çökelir)
% DAVRANIŞI:
%   HS_aq < HS_sat ise: r_prec_raw = 0 (çökelme yok, henüz doymamış)
%   HS_aq > HS_sat ise: r_prec_raw > 0 (çökelme var)

r_prec     = min(r_prec_raw, Fe_pool);
% AÇIKLAMA: Gerçek çökelme hızı (Fe sınırlaması ile)
# FORMÜL: r_prec = min(istenen_hız, mevcut_Fe)
%   1:1 stoichiometry: 1 Fe + 1 HS -> 1 FeS
#   Eğer Fe az ise: r_prec = Fe_pool (tüm Fe tükenir)
%   Eğer Fe bol ise: r_prec = r_prec_raw (HS sınırlar)
% ÖRNEK:
%   r_prec_raw = 0.5 mmol/L/gün, Fe_pool = 0.2 mmol/L
%   r_prec = min(0.5, 0.2) = 0.2 mmol/L/gün (Fe sınırlı)

% ============================================================================
% 17. GYPSUM ÇÖZÜNMESI
% ============================================================================

% --------------------  dissolution source --------------------
SO4_sat    = env.SO4_sat_gyp;                % mM
% AÇIKLAMA: Doygun SO4 seviyesi (env'den al)

r_diss_gyp = k_diss_gyp * max(0, SO4_sat - SO4); % mM/day
% AÇIKLAMA: Gypsum çözünme hızı (mmol/L/gün)
% FORMÜL: r_diss = k_diss * max(0, SO4_sat - SO4)
%   k_diss_gyp: Çözünme hız katsayısı (1/gün)
%   SO4_sat - SO4: Doygunluğa uzaklık
#   max(0, ...): Sadece SO4 < SO4_sat ise çözünür
% DAVRANIŞI:
%   SO4 < SO4_sat ise: Gypsum çözünür, SO4 artar
%   SO4 >= SO4_sat ise: r_diss = 0 (çözünme durur, denge)

% ============================================================================
% 18. GAZ FAZENDEKİ TÜREVLER (dnH2_g/dt, ...)
% ============================================================================

% -------------------- Gas balances (mmol/day) --------------------
dnH2_g  = - J_H2  * Vl;
% AÇIKLAMA: Headspace H2'nin değişim hızı (mmol/gün)
% FORMÜL: dnH2_g/dt = - J_H2 * Vl
%   J_H2: Sıvı fazındaki transfer hızı (mmol/L/gün)
%   Vl: Sıvı hacmi (L)
%   Çarpım: mmol/gün (extensive, toplam mol değişimi)
% İŞARET:
%   J_H2 > 0: Sıvıya giriyor -> Gazdan azalıyor (dnH2_g < 0) ✓
%   J_H2 < 0: Sıvıdan çıkıyor -> Gaza ekleniyor (dnH2_g > 0)

dnCO2_g = - J_CO2 * Vl;
% AÇIKLAMA: Aynı mantık CO2 için

dnCH4_g = + r_meth * Vl;   % CH4 to gas
% AÇIKLAMA: Headspace CH4'ün değişim hızı
# FORMÜL: dnCH4_g/dt = + r_meth * Vl
%   r_meth: Methanogenesis hızı (mmol/L/gün) - sıvı fazda üretilen CH4
%   Vl: Sıvı hacmi (L)
%   Çarpım: mmol/gün
% İŞARET:
%   + : CH4 üretilir (sıvıda oluşur, gaza geçer)
% NOT: Instant degassing varsayımı (CH4 hemen gaza geçer, dissolved CH4 yok)

dnH2S_g = + Jout_H2S * Vl; % degassing of H2S(aq)
% AÇIKLAMA: Headspace H2S'nin değişim hızı
% FORMÜL: dnH2S_g/dt = + Jout_H2S * Vl
%   Jout_H2S: H2S çıkış hızı (mmol/L/gün)
% İŞARET:
%   + : Sıvıdan gaza gidiyor (degassing)

% ============================================================================
% 19. SIVI FAZINDAKİ TÜREVLER (dH2_aq/dt, ...)
% ============================================================================

% -------------------- Differential equations --------------------
% Liquid balances (mmol/L/day)
dH2_aq  = + J_H2  - 4*r_meth - 4*r_sulf - 4*r_aceto;
% AÇIKLAMA: Sıvı fazdaki H2 konsantrasyonunun türevi (mmol/L/gün)
# TERIMLERIN AÇIKLAMASI:
%   + J_H2: Gazdan sıvıya transfer (pozitif = artış)
%   - 4*r_meth: Methanogenesis tüketimi (4H2 kullanılır)
%   - 4*r_sulf: Sulfate reduction tüketimi (4H2 kullanılır)
%   - 4*r_aceto: Acetogenesis tüketimi (4H2 kullanılır)
% STOİKYOMETRİ:
%   4H2 + CO2 → CH4 + 2H2O (methanogenesis)
%   4H2 + SO4 → H2S + 4OH⁻ (sulfate reduction)
%   4H2 + 2CO2 → Acetate + ... (acetogenesis)
% NET ETKİ:
%   Eğer J_H2 > 4*(r_meth+r_sulf+r_aceto) ise: dH2_aq > 0 (H2 artıyor)
%   Aksi halde: dH2_aq < 0 (H2 azalıyor)

dCO2_aq = + J_CO2 - 1*r_meth - 2*r_aceto;   % hydrogenotrophic SR does not consume CO2
% AÇIKLAMA: Sıvı fazdaki CO2 türevi
% TERIMLERIN AÇIKLAMASI:
%   + J_CO2: Gazdan sıvıya transfer
%   - 1*r_meth: Methanogenesis 1CO2 tüketir
%   - 2*r_aceto: Acetogenesis 2CO2 tüketir
# DİKKAT: r_sulf YOK! (SR CO2 tüketmez, yorum satırında belirtilmiş)

dSO4    = - 1*r_sulf + r_diss_gyp;
% AÇIKLAMA: Sülfat konsantrasyonunun türevi
%   - r_sulf: SR tarafından tüketilir (1:1 stoichiometry)
%   + r_diss_gyp: Gypsum çözünmesinden gelir (kaynak)

dFeS    = + r_prec;
% AÇIKLAMA: FeS çökelme miktarının türevi
%   + r_prec: Çökelme hızı (artış)
% NOT: Çözünme yok (tek yönlü reaksiyon)

dX      = + Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X;
% AÇIKLAMA: Biomass türevi
% TERIMLERIN AÇIKLAMASI:
%   + Y_m*r_meth: Methanogenesis'ten büyüme (yield * hız)
%   + Y_s*r_sulf: SR'den büyüme
%   + Y_a*r_aceto: Acetogenesis'ten büyüme
%   - b*X: Doğal ölüm/bozunma (birinci dereceden, decay)
# FORMÜL: dX/dt = Σ(Y_i * r_i) - b*X
%   Yield (Y): Her birim substrat başına ne kadar biomass
%   ÖRNEK: Y_m=0.06 -> 1 mmol CH4 üretimi = 0.06 mmol biomass artışı

dAc     = + r_aceto;
% AÇIKLAMA: Acetate türevi
%   + r_aceto: Acetogenesis tarafından üretilir
% NOT: Tüketim yok (accumulation)

dHCO3   = 0;
% AÇIKLAMA: Bicarbonate SABİT (değişmiyor)
# BU MODELDE: HCO3 dynamic değil, sabit kabul edilmiş

% ============================================================================
% 20. FE POOL TÜREVI
% ============================================================================

% Fe pool balance (mmol/L/day)
dFe_pool = - r_prec;
% AÇIKLAMA: Çözünmüş Fe²⁺ havuzunun türevi
%   - r_prec: FeS çökelmesi Fe tüketir (1:1 stoichiometry)
# MANTIK: Fe + HS -> FeS (aq) (çökelek)
%         Fe azalır, FeS artar
% SINIRLAMA: Fe_pool >= 0 olmalı (guards ile korunuyor)

% ============================================================================
% 21. SÜLFÜR DENGE DENKLEMİ
% ============================================================================

% Total dissolved sulfide balance
dS_tot = + 1.00*r_sulf - r_prec - Jout_H2S;
% AÇIKLAMA: Toplam çözünmüş sülfür (H2S(aq) + HS⁻) türevi
% TERIMLERIN AÇIKLAMASI:
%   + r_sulf: SR tarafından üretilir (1:1 stoichiometry)
%   - r_prec: Çökelme ile kaybedilir (HS⁻ çökelir)
%   - Jout_H2S: Degassing ile kaybedilir (H2S gaza geçer)
# SÜLFÜR KORUNUMU:
%   Üretim: SR
%   Kayıp: Çökelme + Degassing

% ============================================================================
% 22. LAG TRAKİNG DEĞİŞKENİ
% ============================================================================

% Lag tracker
dLag = (f_lag - Lag)/max(w_lag,1e-3);
% AÇIKLAMA: Lag variable'ın türevi
# FORMÜL: dLag/dt = (hedef - mevcut) / zaman_sabiti
%   f_lag: Hedef değer (0 veya 1)
%   Lag: Mevcut değer
%   w_lag: Zaman sabiti (ne kadar hızlı takip edecek)
% DAVRANIŞI:
%   Lag < f_lag ise: dLag > 0 (yavaşça artıyor)
%   Lag > f_lag ise: dLag < 0 (yavaşça azalıyor)
# BİRİNCİ DERECEDEN GEÇİCİ YANITLI (First-order lag):
%   Lag(t) → f_lag(t) üssel olarak (τ = w_lag)
% AMAÇ: Ani geçişleri yumuşatmak (smooth transition)

% ============================================================================
% 23. TÜREV VEKTÖRÜNÜ BİRLEŞTİR
% ============================================================================

% Collect derivatives
dydt = [dnH2_g; dnCO2_g; dnCH4_g; dnH2S_g; dH2_aq; dCO2_aq; dSO4; dFeS; dX; dAc; dHCO3; dS_tot; dLag; dFe_pool];
% AÇIKLAMA: Tüm türevleri tek bir vektörde topla
# YAPISI: dydt = [14x1] sütun vektörü
%   dydt(1)  = dnH2_g   : H2 gaz türevi
%   dydt(2)  = dnCO2_g  : CO2 gaz türevi
%   dydt(3)  = dnCH4_g  : CH4 gaz türevi
%   dydt(4)  = dnH2S_g  : H2S gaz türevi
%   dydt(5)  = dH2_aq   : H2 sıvı türevi
%   dydt(6)  = dCO2_aq  : CO2 sıvı türevi
%   dydt(7)  = dSO4     : SO4 türevi
%   dydt(8)  = dFeS     : FeS türevi
%   dydt(9)  = dX       : Biomass türevi
%   dydt(10) = dAc      : Acetate türevi
%   dydt(11) = dHCO3    : Bicarbonate türevi (0)
%   dydt(12) = dS_tot   : Sülfür türevi
%   dydt(13) = dLag     : Lag türevi
%   dydt(14) = dFe_pool : Fe pool türevi

% NOKTALILI VIRGÜL (;) KULLANIMI:
%   [a; b; c]: Sütun vektörü (dikey birleştirme)
%   [a, b, c]: Satır vektörü (yatay birleştirme)
% ODE15S'E DÖNÜŞ:
%   ode15s bu dydt vektörünü alır
%   Sonraki zaman adımını hesaplar: y_next = y + dydt*dt (yaklaşık)
%   Döngü devam eder: t=0 -> t=20 gün

end % ---- end model_mixed function ----
% AÇIKLAMA: Fonksiyon bitişi
% HER ÇAĞRIDA:
%   INPUT: t, y (mevcut zaman ve durum)
%   ÇALIŞMA: 14 diferansiyel denklemi hesapla
%   OUTPUT: dydt (14 türev)
% KULLANIM:
%   ode15s bu fonksiyonu binlerce kere çağırır (her zaman adımında)
%   Sonuç: y(t) = [14 state variable'ın zamanla evrimi]

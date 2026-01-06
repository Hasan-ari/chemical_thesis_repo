% ============================================================================
% PART 4: SİMULASYON VE PLOTTING - ÖZET
% ============================================================================
% Bu kısım çok uzun olduğu için detaylı yorumlanmadı.
% Temel mantığı burada özetliyoruz.
% ============================================================================

%% -------------------- Final simulation (dense grid) --------------------
% AMAÇ: Fitting tamamlandıktan sonra, p_fit ile dense grid simülasyon yap

odes = @(t,y) model_mixed(t,y,p_fit,env);
% AÇIKLAMA: ODE fonksiyonunu p_fit ile oluştur (artık p0 değil!)

opts = odeset('NonNegative',1:14,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
[t_sim, y_sim] = ode15s(odes, [0 t_exp(end)], y0, opts);
% AÇIKLAMA: ODE'yi çöz
% DİKKAT: t_exp değil, [0 t_exp(end)] kullanılmış
%   NEDEN: ode15s kendi zaman adımlarını belirlesin (adaptive)
#   SONUÇ: t_sim daha sık örneklenmiş (örn: 1000 nokta vs 40 nokta)
%   AMAÇ: Pürüzsüz grafikler için

% ============================================================================
% SULFIDE SPECIATION
% ============================================================================

[H2S_aq, HS_aq] = speciate_sulfide(y_sim(:,12), env.pH_fun(t_sim), env.pKa_H2S);
% AÇIKLAMA: S_tot'u H2S ve HS⁻'e ayır (helper function kullanarak)
% INPUT: y_sim(:,12) = S_tot (tüm zaman noktaları)
# OUTPUT: H2S_aq, HS_aq vektörleri

% ============================================================================
% SÜLFÜR KÜTLE DENGESİ DİAGNOSTİĞİ (Satır 193-235)
% ============================================================================
% AMAÇ: Sülfür korunuyor mu kontrol et
%
# HESAPLAMALAR:
%   S_gas_mmol  = y_sim(:,4)              (headspace H2S)
%   S_aq_mmol   = y_sim(:,12) * Vl        (dissolved sulfide)
%   S_FeS_mmol  = y_sim(:,8) * Vl         (precipitated FeS)
%   S_total_model = S_gas + S_aq + S_FeS
%
#   S_prod_cum = cumtrapz(t_sim, r_sulf_vec) * Vl  (produced sulfur)
#   S_total_expected = S_total0 + S_prod_cum
%
# PLOT:
#   Model total S vs Expected total S
#   Mass balance error = Model - Expected
%
# İDEAL DURUM: Error ≈ 0 (sülfür korunuyor)

% ============================================================================
% H2S HEADSPACE DİAGNOSTİĞİ (Satır 236-249)
% ============================================================================
# AMAÇ: nH2S_g (model) ile Henry'den hesaplanan nH2S_g_eq karşılaştır
#
# HESAP:
%   phi_H2S_fit = p_fit(27)
%   Hcp_H2S_eff = phi_H2S_fit * env.Hcp_H2S_eff
%   nH2S_g_eq = (H2S_aq * Vg) / (Hcp * R * T) * 1000
#
# PLOT:
#   nH2S_g (model) vs nH2S_g_eq (from Henry)
#
# İDEAL DURUM: İki eğri örtüşür (equilibrium var)
# SAPMA: kla_H2S veya alpha_H2S yetersiz (transfer kinetiği yavaş)

% ============================================================================
# REAKSİYON HIZLARINI HESAPLA (Satır 252-255)
# ============================================================================

rates = zeros(length(t_sim), 4);
for k = 1:length(t_sim)
    rates(k,:) = rate_out_mixed(t_sim(k), y_sim(k,:), p_fit, env);
end
# AÇIKLAMA: Her zaman noktası için r_meth, r_sulf, r_prec, r_aceto hesapla
# DÖNGÜ:
#   k = 1: t_sim(1), y_sim(1,:) için [r_meth, r_sulf, r_prec, r_aceto]
#   k = 2: t_sim(2), y_sim(2,:) için [...]
%   ...
# SONUÇ: rates = [N x 4] matris

% ============================================================================
# .DAT DOSYASI YAZIMI (Satır 257-279)
% ============================================================================

fileID = fopen('Sandstone_25C_inc_rates.dat','w');
% AÇIKLAMA: Dosya aç (write mode)

fprintf(fileID, ['Time(days) nH2_g nCO2_g ... \n']);
# AÇIKLAMA: Başlık satırı yaz

for i = 1:length(t_sim)
    fprintf(fileID, fmt, t_sim(i), y_sim(i,1), y_sim(i,2), ..., rates(i,1), ...);
end
# AÇIKLAMA: Her satır bir zaman noktası
# FORMAT: Zaman, 14 state, 2 speciation, 4 rate -> toplam 21 sütun

fclose(fileID);
% AÇIKLAMA: Dosyayı kapat

% ============================================================================
# PLOTLARfigure('Name','Gases & Aqueous - Sandstone (25 °C)');
for i = 1:length(species)
    subplot(7,2,i)
    ...
end
% AÇIKLAMA: 14 subplot (her state için bir grafik)
# SUBPLOT YAPISI: 7 satır, 2 sütun
#   (1,1): nH2_g
#   (1,2): nCO2_g
%   ...
#   (7,2): Fe_pool
#
# HER SUBPLOT:
#   - Deneysel veri: 'ko' (black circles)
#   - Model: 'b-' (blue line)

%% Figure 2: Sulfide speciation & pH
# AÇIKLAMA: H2S(aq), HS⁻, pH grafiği (3 subplot)

%% Figure 3: Kinetic rates
plot(t_sim, rates(:,1), 'r-', 'DisplayName','r_{meth}');
plot(t_sim, rates(:,2), 'b-', 'DisplayName','r_{sulf}');
plot(t_sim, rates(:,3), 'g-', 'DisplayName','r_{precip}');
plot(t_sim, rates(:,4), 'm-', 'DisplayName','r_{aceto}');
# AÇIKLAMA: 4 reaksiyon hızını aynı grafikte göster
# RENKLER: red=meth, blue=sulf, green=precip, magenta=aceto

% ============================================================================
% RMSE HESAPLAMA (Satır 320-324)
% ============================================================================

yH2_on_exp  = interp1(t_sim, y_sim(:,1), t_exp, 'linear', 'extrap');
# AÇIKLAMA: Model çıktısını deneysel zaman noktalarına interpolate et
#   NEDEN: y_sim farklı t_sim noktalarında (dense grid)
#          data_exp t_exp noktalarında (sparse)
#          Karşılaştırma için aynı noktalar gerekli
#
# interp1:
#   (t_sim, y_sim(:,1)): Bilinen noktalar (model output)
#   t_exp: İstenen noktalar (experimental times)
#   'linear': Doğrusal interpolasyon
#   'extrap': Sınırların dışında ekstrapolasyon

rmse_equal(yH2_on_exp, data_exp(:,1))
# AÇIKLAMA: Model vs data RMSE hesapla (H2 için)

% ============================================================================
% ÖZETarınlama ve plotting
# - Fitting tamamlandı, artık p_fit var
# - Dense grid simülasyonu: Pürüzsüz grafikler için
# - Diagnostics: Sülfür dengesi, H2S equilibrium
# - Reaksiyon hızları: rate_out_mixed ile hesaplandı
# - Çıktılar:
#   * .dat dosyası (21 sütun, tüm state'ler + hızlar)
#   * 3 figure (14 subplot + speciation + rates)
#   * RMSE değerleri (model performansı)

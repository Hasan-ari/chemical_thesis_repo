% ============================================================================
% PART 3: FİTTİNG - Parametreleri Optimize Et
% ============================================================================
% Bu kısım:
% - lsqnonlin() çağrısı ile parametreleri optimize eder
% - residuals_full() fonksiyonunu minimize eder
% - En iyi parametreleri p_fit değişkenine kaydeder
% - Sonuçları .mat dosyasına yazar
% ============================================================================

%% -------------------- Fit (solve residuals at t_exp) --------------------
% OPTİMİZASYON: Model çıktıları ile deneysel veriler arasındaki farkı minimize et

[p_fit,~,~,~,~,~,~] = lsqnonlin(@(p) residuals_full(p, t_exp, data_exp, y0, env), p0, lb, ub, fit_opts);
% ============================================================================
% AÇIKLAMA: lsqnonlin fonksiyonu parametreleri optimize eder
% AMAÇ: Minimize et -> sum( (model - data)^2 )
% ============================================================================

% ============================================================================
% SATIR SATIR AÇIKLAMA
% ============================================================================

% SOL TARAF: [p_fit,~,~,~,~,~,~]
% --------------------------------
% AÇIKLAMA: lsqnonlin 7 çıktı üretir, biz sadece ilkini (p_fit) kullanıyoruz
% [p_fit, ~, ~, ~, ~, ~, ~] yapısı:
%   p_fit: Optimize edilmiş parametre vektörü [1x28]
%   ~: resnorm (kare hataların toplamı) - kullanmıyoruz, atla
%   ~: residual (hata vektörü) - kullanmıyoruz, atla
%   ~: exitflag (çıkış durumu: 1=başarı, 0=iterasyon limiti, vs) - atla
%   ~: output (detaylı bilgi struct) - atla
%   ~: lambda (Lagrange çarpanları) - atla
%   ~: jacobian (Jacobian matrisi) - atla

% TILDE (~) OPERATÖRÜ:
%   MATLAB'da ~ = "bu çıktıyı görmezden gel, kullanmayacağım"
%   Bellekte yer kaplamaz (optimize edilir)

% SAĞ TARAF: lsqnonlin(...)
% --------------------------

% 1. PARAMETRE: @(p) residuals_full(p, t_exp, data_exp, y0, env)
%    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
%    ANONİM FONKSİYON (Function Handle)
%    --------------------------
%    YAPISI: @(input) fonksiyon_çağrısı(input, sabit1, sabit2, ...)
%
%    @(p): "p" isimli bir input al
%    residuals_full(p, t_exp, data_exp, y0, env): Bu fonksiyonu çağır
%
%    AMAÇ: lsqnonlin sadece TEK PARAMETRE alan fonksiyon bekler
%          Ama residuals_full 5 parametre alır: (p, t_exp, data_exp, y0, env)
%          Çözüm: Anonim fonksiyon ile p dışındakileri "sabitle"
%
%    NASIL ÇALIŞIR:
%      - lsqnonlin içeriden p=[0.06, 0.08, ...] gibi değerler dener
%      - Her denemede: residuals_full(p, t_exp, data_exp, y0, env) çağrılır
%      - t_exp, data_exp, y0, env SABİT kalır (dışarıdan capture edilir)
%      - Sadece p DEĞİŞİR (optimize ediliyor)
%
%    ÖRNEK AÇILIM:
%      Eğer lsqnonlin p=[0.1, 0.2, ...] denerse
%      -> residuals_full([0.1,0.2,...], t_exp, data_exp, y0, env) çağrılır
%      -> Bir hata vektörü döner, örn: [0.5, -0.3, 0.1, ...]
%      -> lsqnonlin bu hatanın karesini minimize etmeye çalışır

% 2. PARAMETRE: p0
%    ^^^^^^^^^^^^^
%    AÇIKLAMA: Başlangıç tahmin vektörü [1x28]
%    İÇERİK: [0.06, 0.08, 0.03, ..., 1.00]
%    AMAÇ: Optimizasyon bu noktadan başlar
%    ÖNEMLİ: İyi başlangıç -> hızlı çözüm, kötü başlangıç -> yerel minimum'da takılma

% 3. PARAMETRE: lb
%    ^^^^^^^^^^^^^
%    AÇIKLAMA: Lower Bound (alt sınır) [1x28]
%    İÇERİK: [1e-4, 1e-4, 1e-4, ..., 0.70]
%    KISIT: p(i) >= lb(i) for all i
%    AMAÇ: Parametrelerin fiziksel olarak anlamsız değerlere gitmesini engelle
%    ÖRNEK: k_m >= 1e-4 (negatif hız sabiti olamaz)

% 4. PARAMETRE: ub
%    ^^^^^^^^^^^^^
%    AÇIKLAMA: Upper Bound (üst sınır) [1x28]
%    İÇERİK: [5, 5, 5, ..., 3.00]
%    KISIT: p(i) <= ub(i) for all i
%    AMAÇ: Parametrelerin aşırı büyük değerlere gitmesini engelle

% 5. PARAMETRE: fit_opts
%    ^^^^^^^^^^^^^^^^^^^^^^
%    AÇIKLAMA: Optimizasyon seçenekleri (options struct)
%    İÇERİK: Display='iter', MaxFunctionEvaluations=6000
%    AMAÇ: Optimizer'ın davranışını kontrol et

% ============================================================================
% ÇALIŞMA AKIŞI (lsqnonlin içinde ne oluyor?)
% ============================================================================
% 1. BAŞLANGIÇ: p = p0 = [0.06, 0.08, 0.03, ...]
%
% 2. İTERASYON 1:
%    a) residuals_full(p, t_exp, data_exp, y0, env) çağır
%       -> İçinde: ode15s çözer, y_sim hesaplar
%       -> model vs data karşılaştır
%       -> res vektörü döndür, örn: [0.5, -0.3, 0.1, ...] (N_data*5 uzunlukta)
%    b) Hata = sum(res.^2) hesapla
%    c) Gradient (eğim) hesapla (Jacobian ile)
%    d) p'yi güncelle (gradient descent benzeri)
%
% 3. İTERASYON 2:
%    a) Yeni p ile tekrar residuals_full çağır
%    b) Hata azaldı mı? Evet -> devam, Hayır -> adım boyutunu küçült
%
% 4. DURDURMA KRİTERİ:
%    - Gradient çok küçük (değişim yok)
%    - Hata değişimi çok küçük
%    - MaxFunctionEvaluations=6000'e ulaşıldı
%    - lb veya ub sınırına çarpıldı
%
% 5. SONUÇ: p_fit = optimize edilmiş parametre vektörü

% ============================================================================
% lsqnonlin SINIR KONTROLÜ
% ============================================================================
% OTOMATIK OLARAK YAPILANLAR:
%   - p(i) < lb(i) ise -> p(i) = lb(i) yap (alta çarpma)
%   - p(i) > ub(i) ise -> p(i) = ub(i) yap (üste çarpma)
%   - Sınırda mıyız? -> gradient'i sınıra paralel yap (trust-region)

% ============================================================================
% ÖRNEK EKRAN ÇIKTISI (Display='iter' nedeniyle)
% ============================================================================
%  Iteration  Func-count     f(x)          Step-size       optimality
%      0          29         1234.56                        567.8
%      1          58         987.65        0.123            234.5
%      2          87         654.32        0.234            123.4
%      ...
%    100        2900          12.34        0.001              0.5
%
% AÇIKLAMA:
%   - Iteration: Kaç iterasyon geçti
%   - Func-count: Kaç kere residuals_full çağrıldı
%   - f(x): Toplam hata (sum(res.^2))
%   - Step-size: Parametre güncellemesinin büyüklüğü
%   - optimality: Gradient norm (0'a yakın = optimumda)

% ============================================================================
% OPTİMİZASYON BİTTİKTEN SONRA
% ============================================================================
% p_fit artık optimize edilmiş parametreleri içerir
% ÖRNEK:
%   p_fit = [0.0532, 0.0912, 0.0278, ..., 0.98, 1.03, 0.95, 1.12]
%            ^^^^^^  ^^^^^^  ^^^^^^       ^^^^  ^^^^  ^^^^  ^^^^
%             k_m     k_s     k_a         phi_H2 phi_CO2 phi_H2S alpha_H2S
%
% Başlangıç p0'dan farklı (optimize edildi):
%   p0(1)  = 0.06   -> p_fit(1)  = 0.0532  (k_m biraz azaldı)
%   p0(26) = 1.00   -> p_fit(26) = 1.03    (phi_CO2 biraz arttı)

% ============================================================================
% SONUÇLARI KAYDET
% ============================================================================

save('best_fit_params_Sandstone_25C.mat','p_fit','env','y0');
% AÇIKLAMA: .mat dosyasına değişkenleri kaydet
% MATLAB'IN SAVE KOMUTU:
%   save('dosya_adı.mat', 'değişken1', 'değişken2', ...)

% PARAMETRELER:
%   'best_fit_params_Sandstone_25C.mat': Dosya adı
%     -> KONUM: Çalışma dizininde (pwd)
%     -> FORMAT: MATLAB binary format (.mat)
%     -> İÇERİK: p_fit, env, y0 değişkenlerini içerir
%
%   'p_fit': Optimize edilmiş parametreler [1x28]
%   'env': Çevre değişkenleri struct (Vg, Vl, T, ...)
%   'y0': Başlangıç durumu [1x14]

% NEDEN KAYDET:
%   - Sonraki çalıştırmalarda yüklenmek için
%   - Yeniden fitting yapmadan doğrudan simülasyon çalıştırabilirsin
%   - Farklı senaryoları karşılaştırmak için

% NASIL YÜKLENIR:
%   load('best_fit_params_Sandstone_25C.mat')
%   -> p_fit, env, y0 değişkenleri workspace'e yüklenir

% ============================================================================
% FİTTİNG SÜRESİ
% ============================================================================
% TAHMINI SÜRE:
%   - Her iterasyon: ~0.1-1 saniye (ODE çözümü zaman alır)
%   - Toplam iterasyon: 100-500 (bağlı olarak)
%   - TAHMINI: 1-10 dakika (karmaşık sistemler için saatler olabilir)
%
% HIZLANDIRMA:
%   - ODE toleranslarını gevşet (RelTol, AbsTol)
%   - MaxFunctionEvaluations azalt (ama kötü sonuç riski)
%   - Daha iyi p0 tahmin et (domain bilgisi ile)

% ============================================================================
% ÖNEMLİ NOTLAR
% ============================================================================
% 1. YEREL MİNİMUM SORUNU:
%    - lsqnonlin local optimizer'dır (global değil!)
%    - Farklı p0 ile farklı sonuçlar bulabilir
%    - Çözüm: Multi-start yapabilirsin (farklı p0'larla birkaç kez çalıştır)
%
% 2. HATA FONKSİYONU:
%    - residuals_full içinde log1p(x) transformu var
%    - Ağırlıklar var: [1, 1, 0.9, 1.0, 2.0]
%    - Bu sebeple farklı verilere farklı önem veriliyor
%
% 3. ODE ÇÖZÜMÜ:
%    - Her iterasyonda ode15s çağrılır (yavaş!)
%    - NonNegative kısıtı var (state'ler negatif olamaz)
%    - Eğer ODE patlıyorsa (solver fails) -> res'e ceza ekle

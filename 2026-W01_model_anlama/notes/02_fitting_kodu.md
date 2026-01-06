# Fitting Kısmı - Satır Satır Kod Açıklaması

**Bölüm**: Satır 114-116 (Fitting) + 258-276 (residuals_full fonksiyonu)
**Amaç**: Parametreleri optimize etmek

---

## Satır 114: lsqnonlin Çağrısı

```matlab
[p_fit,~,~,~,~,~,~] = lsqnonlin(@(p) residuals_full(p, t_exp, data_exp, y0, env), p0, lb, ub, fit_opts);
```

**Parçalara ayıralım:**

### Sol Taraf (Output)
```matlab
[p_fit,~,~,~,~,~,~] = ...
```
- `p_fit` → En iyi parametreler (28×1 vector)
- `~,~,~,~,~,~` → Diğer output'ları kullanma (ignore et)

### lsqnonlin Fonksiyonu
```matlab
lsqnonlin(@(p) residuals_full(p, t_exp, data_exp, y0, env), p0, lb, ub, fit_opts)
```

**Parametreler:**
1. `@(p) residuals_full(...)` → Objective function
   - `@(p)` → Anonymous function (input: `p`)
   - `residuals_full(p, t_exp, data_exp, y0, env)` → Residual hesapla
2. `p0` → Initial guess (28 parametre)
3. `lb` → Lower bounds
4. `ub` → Upper bounds
5. `fit_opts` → Options (max iterations, display, etc.)

**Ne yapıyor?**
```
1. p = p0 ile başla
2. residuals = residuals_full(p, ...) hesapla
3. Sum of squares: SSR = sum(residuals.^2)
4. SSR minimize et (p değiştir)
5. Converge olunca p_fit döndür
```

---

## residuals_full Fonksiyonu (Satır 258-276)

```matlab
function res = residuals_full(p, t_exp, data_exp, y0, env)
```

**Input:**
- `p`: 28 parametre (deneniyor, optimizer verir)
- `t_exp`: Deneysel zaman noktaları (örnek: 20 nokta)
- `data_exp`: Deneysel data (20×5 matrix)
- `y0`: Başlangıç koşulu (14 state)
- `env`: Sabitler struct

**Output:**
- `res`: Residual vector (flattened)

---

## Satır 260-261: ODE Çöz

```matlab
odes = @(t,y) model_mixed(t,y,p,env);
opts = odeset('NonNegative',1:14,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
[~, y_sim] = ode15s(odes, t_exp, y0, opts);
```

**Satır satır:**

### Satır 260: ODE Function Handle
```matlab
odes = @(t,y) model_mixed(t,y,p,env);
```
- Anonymous function oluştur
- Input: `(t, y)` → Output: `dydt`
- `p` ve `env` "frozen" (closure)

### Satır 261: ODE Options
```matlab
opts = odeset('NonNegative',1:14, 'RelTol',1e-8, 'AbsTol',1e-10, 'MaxStep',0.5);
```
- `'NonNegative',1:14` → Tüm 14 state ≥ 0 zorla
- `'RelTol',1e-8` → Relative tolerance (0.00000001)
- `'AbsTol',1e-10` → Absolute tolerance
- `'MaxStep',0.5` → Max timestep = 0.5 gün

### Satır 262: Solver Çağrısı
```matlab
[~, y_sim] = ode15s(odes, t_exp, y0, opts);
```
- `ode15s` → Stiff ODE solver
- `t_exp` → Sadece bu noktalarda çöz (örnek: [0, 1, 2, ..., 20])
- `y0` → Başlangıç (14×1)
- Output:
  - `~` → t vektörünü ignore et (zaten `t_exp`)
  - `y_sim` → 20×14 matrix (her satır bir timestep)

**Örnek:**
```
t_exp = [0, 1, 2, 3, 4, 5]  (6 nokta)
y_sim = [
  [nH2_g(0), nCO2_g(0), ..., Fe_pool(0)],  % t=0
  [nH2_g(1), nCO2_g(1), ..., Fe_pool(1)],  % t=1
  ...
  [nH2_g(5), nCO2_g(5), ..., Fe_pool(5)]   % t=5
]
```

---

## Satır 264-266: Model vs Data Karşılaştırması

```matlab
sim_mat = [y_sim(:,1), y_sim(:,2), y_sim(:,3), y_sim(:,4), y_sim(:,7)];
log_sim = log1p(sim_mat);
log_exp = log1p([data_exp(:,1), data_exp(:,2), data_exp(:,3), data_exp(:,4), data_exp(:,5)]);
```

**Satır satır:**

### Satır 264: Model Çıktısını Seç
```matlab
sim_mat = [y_sim(:,1), y_sim(:,2), y_sim(:,3), y_sim(:,4), y_sim(:,7)];
```
- `y_sim(:,1)` → nH2_g (model)
- `y_sim(:,2)` → nCO2_g
- `y_sim(:,3)` → nCH4_g
- `y_sim(:,4)` → nH2S_g
- `y_sim(:,7)` → SO4
- `sim_mat`: 20×5 matrix

**Neden bu 5 state?**
- Bunlar deneysel olarak ölçülen değerler
- `data_exp` ile aynı boyut olmalı

### Satır 265: Log Transform (Model)
```matlab
log_sim = log1p(sim_mat);
```
- `log1p(x) = log(1 + x)` → Güvenli log
- Neden? Eğer `x = 0` → `log(0) = -Inf` (sorun!)
- `log1p(0) = log(1) = 0` (güvenli)

### Satır 266: Log Transform (Data)
```matlab
log_exp = log1p([data_exp(:,1), data_exp(:,2), data_exp(:,3), data_exp(:,4), data_exp(:,5)]);
```
- Aynı transform'u data'ya uygula
- `data_exp`: 20×5 (5 kolon: H2, CO2, CH4, H2S, SO4)

---

## Satır 269-270: Weighted Residuals

```matlab
weights = [1, 1, 0.9, 1.0, 2.0];
res = (log_sim - log_exp) .* weights;
```

**Satır satır:**

### Satır 269: Ağırlıklar
```matlab
weights = [1, 1, 0.9, 1.0, 2.0];
```
- `weights(1)` = 1.0 → H2 (normal)
- `weights(2)` = 1.0 → CO2 (normal)
- `weights(3)` = 0.9 → CH4 (biraz daha az önemli)
- `weights(4)` = 1.0 → H2S (normal)
- `weights(5)` = 2.0 → SO4 (**çok önemli!**)

**Neden ağırlık?**
- SO4 plateau'ya ulaşmalı → Fit kalitesi için kritik
- 2× daha fazla penalty

### Satır 270: Residual Hesaplama
```matlab
res = (log_sim - log_exp) .* weights;
```
- `log_sim - log_exp` → Element-wise fark (20×5)
- `.* weights` → Element-wise çarpım
  - `weights` = 1×5 → MATLAB broadcast eder
  - Her kolon için farklı weight

**Örnek:**
```matlab
log_sim = [2.5, 3.0, 1.5, 0.5, 2.0]  (bir satır)
log_exp = [2.3, 2.9, 1.6, 0.6, 2.2]
fark    = [0.2, 0.1, -0.1, -0.1, -0.2]
weights = [1, 1, 0.9, 1.0, 2.0]
res     = [0.2, 0.1, -0.09, -0.1, -0.4]  ← SO4 farkı 2× amplified
```

---

## Satır 273-274: Penalty for Negative States

```matlab
if any(y_sim(:) < -1e-9)
    res = res + 1e3 * abs(min(y_sim(:)));
end
```

**Satır satır:**

### Satır 273: Check Negatif Değer
```matlab
if any(y_sim(:) < -1e-9)
```
- `y_sim(:)` → Tüm elemanları flatten et (20×14 → 280×1)
- `< -1e-9` → -0.000000001'den küçük mü?
- `any(...)` → En az 1 tane varsa `true`

**Neden -1e-9?**
- Numerical error yüzünden çok küçük negatif (örnek: -1e-15) olabilir
- Bunları ignore et, ama gerçek negatif (-0.001) penalize et

### Satır 274: Büyük Penalty Ekle
```matlab
res = res + 1e3 * abs(min(y_sim(:)));
```
- `min(y_sim(:))` → En negatif değer (örnek: -0.05)
- `abs(...)` → Mutlak değer (0.05)
- `1e3 *` → 1000× çarp (50)
- `res = res + 50` → Tüm residual'lara ekle

**Neden?**
- Eğer state < 0 → Fiziksel olarak invalid
- Optimizer bu parametreyi kullanmasın (büyük penalty)

---

## Satır 275: Reshape to Vector

```matlab
res = res(:);
```

**Ne yapıyor?**
- `res` şu an 20×5 matrix
- `res(:)` → 100×1 vector'e çevir (flatten)

**Neden?**
- `lsqnonlin` vector bekliyor, matrix değil
- Sum of squares: `SSR = sum(res.^2)` → Tek sayı

---

## Fitting Algoritması Akışı

```
1. lsqnonlin başlar, p = p0
2. residuals_full(p, ...) çağır
   ├─ ODE çöz: ode15s(odes, t_exp, y0, opts)
   ├─ Model çıktısı: y_sim (20×14)
   ├─ 5 state seç: sim_mat (20×5)
   ├─ Log transform: log_sim, log_exp
   ├─ Weighted residuals: res = (log_sim - log_exp) .* weights
   ├─ Penalty (eğer state < 0)
   └─ Flatten: res → 100×1 vector
3. SSR = sum(res.^2) hesapla
4. Gradient hesapla (finite difference)
5. p güncelle (trust-region algorithm)
6. Converge? Hayır → 2'ye dön
7. Converge? Evet → p_fit döndür
```

---

## Örnek İterasyon

```
Iteration 0:
  p = p0 = [0.06, 0.08, 0.03, ...]
  res = residuals_full(p0, ...)
  SSR = 123.456

Iteration 1:
  p = [0.062, 0.082, 0.031, ...]  (küçük değişiklik)
  res = residuals_full(p, ...)
  SSR = 89.123  (azaldı ✅)

Iteration 2:
  p = [0.064, 0.084, 0.032, ...]
  SSR = 65.432

...

Iteration 150:
  p = p_fit = [0.0631, 0.0847, 0.0289, ...]
  SSR = 1.234  (converged)
```

---

## Satır 115: Save Fitted Parameters

```matlab
save('best_fit_params_Sandstone_25C.mat','p_fit','env','y0');
```

**Ne yapıyor?**
- `save(...)` → .mat dosyasına yaz (MATLAB binary)
- `'best_fit_params_Sandstone_25C.mat'` → Dosya adı
- `'p_fit','env','y0'` → Kaydedilecek değişkenler

**Neden kaydet?**
- Sonraki simülasyonlarda kullanmak için
- `load('best_fit_params_Sandstone_25C.mat')` ile geri yükle

---

## Özet: Fitting Kısmının Görevi

```
1. lsqnonlin çağır
   - Input: residuals_full fonksiyonu, p0, bounds, options
   - Output: p_fit (28 parametre)

2. residuals_full içinde:
   - ODE çöz (ode15s, sadece t_exp noktalarında)
   - Model çıktısını data ile karşılaştır
   - Log transform (küçük-büyük değer dengeleme)
   - Weighted residuals (SO4 daha önemli)
   - Penalty (negatif state'ler için)
   - Vector döndür (lsqnonlin için)

3. Optimizer:
   - Sum of squares minimize et
   - Iteratif olarak p güncelle
   - Converge olunca p_fit döndür

4. Kaydet:
   - p_fit, env, y0 → .mat dosyası
```

**Sonraki adım:** p_fit ile final simülasyon (dense grid)

---

**Hazırlayan**: Hasan Arı
**Tarih**: 6 Ocak 2026
**Amaç**: Fitting kodunu satır satır anlamak

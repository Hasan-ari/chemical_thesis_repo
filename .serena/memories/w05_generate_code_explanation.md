# generate.py - Satır Satır Kod Açıklaması

## Dosya Lokasyonu
`2026_W04_Lstm_training_v1/new_app/src/lstm_synth_data/generate.py`

---

## Ana Fonksiyon: `generate_synthetic_data()`

### 1️⃣ Parametre Yükleme (Satır 47)
```python
p_fit, env, y0 = load_basalt_25c_params(mat_file)
```

| Değişken | Açıklama | Detay |
|----------|----------|-------|
| `p_fit` | 28 fitted parametre | Kimya hocasının MATLAB best-fit'i |
| `env` | Environment config | Vg, Vl, T, Henry sabitleri, pH fonksiyonu |
| `y0` | Başlangıç koşulları | 14 elemanlı vektör, deneysel veriden |

---

### 2️⃣ Uniform Zaman Grid'i Oluştur (Satır 55-56)
```python
t_eval = np.linspace(t_start, t_end, n_points)
dt = t_eval[1] - t_eval[0]
```

| Değişken | Değer | Açıklama |
|----------|-------|----------|
| `t_start` | 0 | Başlangıç zamanı (gün) |
| `t_end` | 19 | Bitiş zamanı (gün) |
| `n_points` | 500 | İstenen nokta sayısı |
| `t_eval` | [0, 0.038, 0.076, ..., 19] | 500 eşit aralıklı nokta |
| `dt` | 19/499 ≈ 0.038 gün | ≈ 55 dakika |

---

### 3️⃣ ODE Fonksiyonunu Tanımla (Satır 64-65)
```python
def ode_func(t, y):
    return model_mixed(t, y, p_fit, env)
```

- `model_mixed` → Kimyasal ODE modeli (`ode_model.py` içinde)
- Input: `t` (zaman), `y` (14 state değişkeni)
- Output: `dydt` (14 türev değeri)

Bu fonksiyon şu diferansiyel denklemi temsil ediyor:
```
dy/dt = f(t, y, p_fit, env)
```

---

### 4️⃣ ODE'yi Çöz (Satır 68-77) - ANA KISIM
```python
sol = solve_ivp(
    ode_func,                    # Çözülecek ODE: dy/dt = f(t, y)
    t_span=(t_start, t_end),     # Zaman aralığı: (0, 19)
    y0=y0,                       # Başlangıç: [9.074, 2.464, 0, ...]
    method='Radau',              # Stiff solver (MATLAB ode15s gibi)
    t_eval=t_eval,               # Bu noktalarda çıktı ver
    rtol=1e-8,                   # Relative tolerance
    atol=1e-10,                  # Absolute tolerance
    max_step=0.5                 # Max adım büyüklüğü (gün)
)
```

#### Parametre Açıklamaları:

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `ode_func` | fonksiyon | dy/dt = f(t, y) |
| `t_span` | (0, 19) | Çözüm aralığı |
| `y0` | 14 elemanlı array | Başlangıç koşulu |
| `method` | 'Radau' | Stiff systems için (ode15s eşdeğeri) |
| `t_eval` | 500 noktalı array | Çıktı istenen zamanlar |
| `rtol` | 1e-8 | Relative tolerance |
| `atol` | 1e-10 | Absolute tolerance |
| `max_step` | 0.5 | Maksimum adım (gün) |

#### solve_ivp Nasıl Çalışıyor?

```
Solver KENDI adımlarıyla ilerliyor (adaptive):
t=0 → t=0.001 → t=0.005 → t=0.02 → ... → t=19
      (küçük)    (küçük)   (büyük)

AMA biz t_eval verdik:
t_eval = [0, 0.038, 0.076, 0.114, ..., 19]

Solver ne yapıyor?
1. Kendi adaptive step'leriyle tam çözümü hesaplar
2. t_eval noktalarında İNTERPOLASYON yapar (dense output)
3. Sadece t_eval noktalarındaki değerleri döndürür
```

---

### 5️⃣ Sonuçları Çıkar (Satır 87-88)
```python
t_sim = sol.t      # Zaman array'i: [0, 0.038, ..., 19]
y_sim = sol.y.T    # State'ler: (500, 14) shape
```

| Değişken | Shape | Açıklama |
|----------|-------|----------|
| `sol.t` | (500,) | Zaman noktaları |
| `sol.y` | (14, 500) | Solver'ın raw çıktısı |
| `sol.y.T` | (500, 14) | Transpose - bizim istediğimiz format |

---

### 6️⃣ Ekstra Hesaplamalar (Satır 91-97)
```python
# pH değerleri (deneysel veriden interpolasyon)
pH_vals = np.array([env.pH_fun(t) for t in t_sim])

# Sülfür spesiyasyonu: S_tot → H2S_aq + HS_aq
H2S_aq, HS_aq = speciate_sulfide(y_sim[:, 11], pH_vals, env.pKa_H2S)

# Reaksiyon hızları (her nokta için)
rates = np.zeros((len(t_sim), 4))
for i in range(len(t_sim)):
    rates[i] = compute_rates(t_sim[i], y_sim[i], p_fit, env)
```

| Hesaplama | Açıklama |
|-----------|----------|
| `pH_vals` | Deneysel pH verisinden interpolasyon |
| `H2S_aq, HS_aq` | Toplam sülfürün (S_tot) pH'a göre dağılımı |
| `rates` | 4 reaksiyon hızı: metanojenez, sülfat red., FeS çökelme, asetojenez |

---

### 7️⃣ Return (Satır 105-117)
```python
return {
    'time': t_sim,        # (500,) - zaman
    'states': y_sim,      # (500, 14) - 14 state değişkeni
    'H2S_aq': H2S_aq,     # (500,) - çözünmüş H2S
    'HS_aq': HS_aq,       # (500,) - çözünmüş HS-
    'rates': rates,       # (500, 4) - reaksiyon hızları
    'pH': pH_vals,        # (500,) - pH değerleri
    'params': p_fit,      # (28,) - parametreler
    'env': env,           # Environment objesi
    'y0': y0,             # (14,) - başlangıç koşulu
    'dt': dt,             # skalar - zaman adımı
    'n_points': n_points  # skalar - nokta sayısı
}
```

---

## Özet Akış Diyagramı

```
┌─────────────────────────────────────────────────────────────┐
│ Satır 47:  .mat dosyası → p_fit, env, y0                    │
│            (28 parametre, environment, başlangıç koşulu)    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Satır 55:  t_eval = np.linspace(0, 19, 500)                 │
│            → [0, 0.038, 0.076, ..., 19]  (500 uniform nokta)│
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Satır 64:  ode_func(t, y) = model_mixed(t, y, p_fit, env)   │
│            → dy/dt hesaplayan fonksiyon                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Satır 68:  sol = solve_ivp(ode_func, t_eval=t_eval, ...)    │
│            → Solver çalışıyor                               │
│            → Adaptive step + dense output interpolation     │
│            → t_eval noktalarında çıktı                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Satır 88:  y_sim = sol.y.T                                  │
│            → Shape: (500, 14)                               │
│            → 500 zaman noktası × 14 state değişkeni         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Satır 166-168 (save_data):                                  │
│   np.save(lstm_path, data['states'])                        │
│   → basalt_25c_lstm_input_500pts.npy                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Neden Bu Yaklaşım?

### Alternatif: MATLAB çıktısını interpolasyon
```
MATLAB .dat (1433 nokta, non-uniform) → interpolasyon → 500 uniform
```

### Bizim yaklaşım: ODE'yi tekrar çöz
```
p_fit + y0 → Python solve_ivp(t_eval=500 nokta) → 500 uniform
```

### Avantajları:
1. ✅ Interpolasyon hatasından kaçınıyoruz
2. ✅ İstediğimiz kadar nokta üretebiliyoruz (500, 1000, 2500)
3. ✅ .dat dosya formatına bağımlı değiliz
4. ✅ Aynı parametrelerle farklı zaman aralıkları deneyebiliriz

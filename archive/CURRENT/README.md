# CURRENT - Şu An Çalışılacak Materyaller

⭐ **Bu klasördeki dosyalar 3 Ocak 2026 akşam mailinden (107 dosya)**

## Üzerinde çalışacak olduğumuz data.

### v4 İki Fazlı Model (Two-Phase)
**Kod Versiyonu**: v4 (en güncel)

**Özellikler**:
- 14 state variables (v3'te 13'tü)
- 28 parameters (v3'te 13'tü)
- İki fazlı (gaz + sıvı) sistem
- Henry yasası implementasyonu
- pH bağımlı sülfür türlenmesi
- Fe pool limitation (Gypsum için kritik)

### 📊 12 Başarılı Fit

| Kayaç | 25°C | 34°C | 40°C |
|-------|------|------|------|
| **Sandstone** | ✅ | ✅ | ✅ |
| **Basalt** | ✅ | ✅ | ✅ |
| **Calcite** | ✅ | ✅ | ✅ |
| **Gypsum** | ✅ | ✅ | ✅ |

**Her durum için** (örnek: `code/v4_two_phase/sandstone_25C/`):
- `anaerobic_model_two_phase_mixedSR_25C_v4.m` - v4 kodu
- `best_fit_params_Sandstone_25C.mat` - Fitted parametreler
- Deneysel veri: `data/muller_2024/Muller_2024_H2_Sandstone_at_25C.txt`
- Sonuçlar: `results/fitted_outputs/` klasöründe

---

## ✅ VERİ DOĞRULAMA (Her çalışma seansı başında!)

### MATLAB'da Çalışırken

```matlab
% 1. CURRENT/ klasöründe olduğundan emin ol
pwd  % Çıktı: .../chemical_thesis_repo/CURRENT olmalı

% 2. Doğru dosyayı kullandığını kontrol et
% ✅ DOĞRU:
edit code/v4_two_phase/sandstone_25C/anaerobic_model_two_phase_mixedSR_25C_v4.m

% ❌ YANLIŞ (archive'deki eski dosyalar):
% ../archive/reactions_old/Matlab codes/rnn_transport_multiguild_uq_v3.m
```

### Python'da Çalışırken

```python
import os

# 1. Ana repo klasöründe olmalısın
print(os.getcwd())  # .../chemical_thesis_repo çıkmalı

# 2. CURRENT/ klasöründeki verileri kullan
# ✅ DOĞRU:
data_path = "CURRENT/data/muller_2024/Muller_2024_H2_Sandstone_at_25C.txt"

# ❌ YANLIŞ:
# "reactions/..."  → Bu klasör artık archive'de
# "archive/..."    → Eski, yanlış veriler
# "src/data/..."   → Bu yapı artık kullanılmıyor
```

### Dosya Varlık Kontrolü

```bash
# Bu dosyalar OLMALI:
ls CURRENT/code/v4_two_phase/sandstone_25C/
# → anaerobic_model_two_phase_mixedSR_25C_v4.m ✅
# → best_fit_params_Sandstone_25C.mat ✅

# Bu dosyalar OLMAMALI (çünkü archive'de):
ls archive/reactions_old/Matlab codes/
# → rnn_transport_multiguild_uq_v3.m (eski v3 kodu)
# → trained_LSTM_multiguild.mat (eski LSTM)
```

---

## HAFTALIK ÇALIŞMA YOL HARİTASI

### Model Anlama 

**Hedef**: v4 iki fazlı modelini tam anlamak

#### Kod Yapısını Anla

```matlab
% 1. Sandstone 25°C kodunu aç
cd CURRENT/code/v4_two_phase/sandstone_25C
edit anaerobic_model_two_phase_mixedSR_25C_v4.m

% 2. Kod bölümlerini belirle ve comment'lerle işaretle:
%    A. State variables tanımı (14 tane)
%       - H2_aq, H2_gas, CH4_aq, CH4_gas, ...
%
%    B. Parameter tanımı (28 tane)
%       - Kinetik sabitler (k1, k2, ...)
%       - Stokiyometrik katsayılar
%
%    C. ODE sistem fonksiyonu
%       - dy/dt denklemleri
%
%    D. Henry yasası hesaplamaları
%       - Gaz-sıvı dengesi
%
%    E. pH speciation formülleri
%       - H2S/HS-/S2- dengesi
```

**Çıktı**: Kodu anladığını gösteren comment'li bir kopya

---

#### Parametreleri İncele

```matlab
% Fitted parametreleri yükle
load('best_fit_params_Sandstone_25C.mat')

% Parametreleri listele ve Excel'e kaydet
param_names = {'k_meth', 'k_sulf', 'K_H2', 'K_CH4', ...};  % 28 parametre
param_values = best_params;  % Loaded values

% Her parametrenin anlamını yaz:
% - k_meth: Methanogenesis rate constant
% - k_sulf: Sulfate reduction rate constant
% - K_H2: H2 half-saturation constant
% ...

% Karşılaştırma için tablo oluştur
T = table(param_names', param_values', 'VariableNames', {'Parameter', 'Value'});
writetable(T, 'parameter_analysis.xlsx');
```

**Çıktı**: Yeni klasör oluştur: `../../../HAFTA_1/`
- `HAFTA_1/sandstone_25C_parameters.xlsx`
- `HAFTA_1/parameter_notes.md` (her parametrenin açıklaması)

---

#### 📌 Modeli Çalıştır ve Analiz Et

```matlab
% Kodu çalıştır
anaerobic_model_two_phase_mixedSR_25C_v4

% Çıkan figürleri kaydet
saveas(gcf, '../../../HAFTA_1/sandstone_25C_H2_production.png');

% Analiz soruları:
% 1. H2 konsantrasyonu zamanla nasıl değişiyor?
% 2. Hangi reaksiyon (methanogenesis vs sulfate reduction) dominant?
% 3. Fitted eğri ile deneysel veri arasında fark var mı? Nerede?
% 4. pH nasıl değişiyor? Bu H2S speciation'ı nasıl etkiliyor?
```

**Çıktı**:
- `HAFTA_1/sandstone_25C_results.png`
- `HAFTA_1/model_analysis.md` (yukarıdaki soruların cevapları)

---

#### Hafta Sonu: Diğer Kayaçları Karşılaştır

```matlab
% Aynı sıcaklıkta (25°C) farklı kayaçları karşılaştır
rocks = {'Sandstone', 'Basalt', 'Calcite', 'Gypsum'};

figure('Position', [100, 100, 1200, 800]);
for i = 1:4
    subplot(2, 2, i);
    cd(sprintf('../%s_25C', lower(rocks{i})));
    anaerobic_model_two_phase_mixedSR_25C_v4;
    title(rocks{i});
    cd ..;
end
saveas(gcf, '../../../HAFTA_1/rock_comparison_25C.png');

% Karşılaştırma soruları:
% 1. Hangi kayaçta H2 üretimi en hızlı?
% 2. Hangi kayaçta methanogenesis dominant?
% 3. Parameter değerleri nasıl farklı? (Excel'de karşılaştır)
```

**Çıktı**:
- `HAFTA_1/rock_comparison_25C.png`
- `HAFTA_1/rock_comparison_notes.md`

---

### 📅 Python Çevirisi

**Hedef**: v4 MATLAB kodunu Python'a çevirmek

#### 📌 Hazırlık: Ortam Kurulumu

```bash
# Ana repo klasöründe
cd d:/chemical_thesis_repo

# Virtual environment oluştur
python -m venv venv_v4
# Windows:
venv_v4\Scripts\activate
# Linux/Mac:
# source venv_v4/bin/activate

# Kütüphaneleri yükle
pip install numpy scipy matplotlib pandas openpyxl

# Çalışma klasörü oluştur
mkdir HAFTA_3
cd HAFTA_3
```

---

#### 📌 ODE Sistemini Python'a Çevir

```python
# anaerobic_model_v4.py

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat

class AnaerobicModelV4:
    """
    v4 Two-Phase Anaerobic H2 Production Model

    14 State Variables:
    0:  H2_aq    - Dissolved H2 (mol/L)
    1:  H2_gas   - Headspace H2 (mol)
    2:  CH4_aq   - Dissolved CH4 (mol/L)
    3:  CH4_gas  - Headspace CH4 (mol)
    4:  H2S_aq   - Dissolved H2S (mol/L)
    5:  H2S_gas  - Headspace H2S (mol)
    6:  SO4      - Sulfate (mol/L)
    7:  FeS      - Iron sulfide (mol)
    8:  X_meth   - Methanogen biomass
    9:  X_sulf   - Sulfate reducer biomass
    10: X_aceto  - Acetogen biomass
    11: Acetate  - Acetate (mol/L)
    12: Fe_pool  - Available iron (mol)
    13: pH       - pH

    28 Parameters:
    [0-27] - Rate constants, saturation constants, stoichiometry
    """

    def __init__(self, params, T=25):
        """
        Args:
            params: 28 model parameters (array)
            T: Temperature in Celsius
        """
        self.params = params
        self.T = T + 273.15  # Kelvin

        # Henry constants (temperature dependent)
        self.H_H2 = self.calculate_henry_H2(T)
        self.H_CH4 = self.calculate_henry_CH4(T)
        self.H_H2S = self.calculate_henry_H2S(T)

    def calculate_henry_H2(self, T_celsius):
        """Henry constant for H2 (mol/L/atm)"""
        T_K = T_celsius + 273.15
        # Van't Hoff equation
        H_298 = 7.8e-4  # at 25°C
        deltaH = -4.5e3  # J/mol
        R = 8.314  # J/(mol*K)
        return H_298 * np.exp(deltaH/R * (1/T_K - 1/298.15))

    def calculate_henry_CH4(self, T_celsius):
        """Henry constant for CH4 (mol/L/atm)"""
        T_K = T_celsius + 273.15
        H_298 = 1.3e-3
        deltaH = -1.7e3
        R = 8.314
        return H_298 * np.exp(deltaH/R * (1/T_K - 1/298.15))

    def calculate_henry_H2S(self, T_celsius):
        """Henry constant for H2S (mol/L/atm)"""
        T_K = T_celsius + 273.15
        H_298 = 1.0e-1
        deltaH = -2.4e3
        R = 8.314
        return H_298 * np.exp(deltaH/R * (1/T_K - 1/298.15))

    def sulfide_speciation(self, pH):
        """
        Calculate sulfide speciation fractions
        H2S <-> HS- <-> S2-

        Returns:
            alpha_H2S, alpha_HS, alpha_S2 (fractions)
        """
        # pKa values
        pKa1 = 7.0  # H2S <-> HS-
        pKa2 = 14.0  # HS- <-> S2-

        H = 10**(-pH)
        Ka1 = 10**(-pKa1)
        Ka2 = 10**(-pKa2)

        denom = H**2 + H*Ka1 + Ka1*Ka2
        alpha_H2S = H**2 / denom
        alpha_HS = (H * Ka1) / denom
        alpha_S2 = (Ka1 * Ka2) / denom

        return alpha_H2S, alpha_HS, alpha_S2

    def ode_system(self, t, y):
        """
        ODE system: dy/dt = f(t, y, params)

        Args:
            t: Time (hours)
            y: State vector (14 variables)

        Returns:
            dydt: Derivative vector (14 equations)
        """
        # Unpack parameters (buraya MATLAB'dan bakarak ekle)
        k_meth = self.params[0]
        k_sulf = self.params[1]
        K_H2 = self.params[2]
        # ... diğer 25 parametre

        # Unpack state variables
        H2_aq, H2_gas, CH4_aq, CH4_gas, H2S_aq, H2S_gas = y[0:6]
        SO4, FeS, X_meth, X_sulf, X_aceto, Acetate, Fe_pool, pH = y[6:14]

        # pH speciation
        alpha_H2S, alpha_HS, alpha_S2 = self.sulfide_speciation(pH)

        # Gas-liquid equilibrium (Henry's law)
        P_H2 = H2_gas / (V_headspace)  # Partial pressure
        H2_eq = self.H_H2 * P_H2

        # Reaction rates
        r_meth = k_meth * (H2_aq / (K_H2 + H2_aq)) * X_meth
        r_sulf = k_sulf * (H2_aq / (K_H2 + H2_aq)) * (SO4 / (K_SO4 + SO4)) * X_sulf
        # ... diğer reaksiyonlar

        # ODE equations (MATLAB'dan çevir)
        dydt = np.zeros(14)

        # 0: dH2_aq/dt
        dydt[0] = -r_meth - r_sulf + k_transfer_H2 * (H2_eq - H2_aq)

        # 1: dH2_gas/dt
        dydt[1] = -k_transfer_H2 * V_liquid * (H2_eq - H2_aq)

        # ... diğer 12 denklem

        return dydt

    def solve(self, t_span, y0, t_eval=None):
        """
        Solve the ODE system

        Args:
            t_span: (t_start, t_end) tuple
            y0: Initial conditions (14 values)
            t_eval: Time points for output (optional)

        Returns:
            sol: ODE solution object
        """
        sol = solve_ivp(
            self.ode_system,
            t_span,
            y0,
            method='BDF',  # Equivalent to MATLAB's ode15s
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9
        )
        return sol

# Test kodu
if __name__ == "__main__":
    # Parametreleri yükle
    params_mat = loadmat('../CURRENT/code/v4_two_phase/sandstone_25C/best_fit_params_Sandstone_25C.mat')
    params = params_mat['best_params'].flatten()

    # Model oluştur
    model = AnaerobicModelV4(params, T=25)

    # Initial conditions (MATLAB'dan kopyala)
    y0 = np.zeros(14)
    y0[6] = 0.02  # SO4 initial
    # ... diğer başlangıç koşulları

    # Çöz
    t_span = (0, 500)  # 500 saat
    t_eval = np.linspace(0, 500, 1000)
    sol = model.solve(t_span, y0, t_eval)

    print(f"Solution successful: {sol.success}")
    print(f"Final H2_aq: {sol.y[0, -1]:.6f} mol/L")
```

**Çıktı**: `HAFTA_3/anaerobic_model_v4.py`

---

#### 📌 Test ve MATLAB Karşılaştırması

```python
# test_vs_matlab.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anaerobic_model_v4 import AnaerobicModelV4
from scipy.io import loadmat

# 1. MATLAB sonuçlarını yükle (eğer .mat file varsa)
# Veya deneysel veriyi yükle
data_exp = pd.read_csv('../CURRENT/data/muller_2024/Muller_2024_H2_Sandstone_at_25C.txt',
                       sep='\t')

# 2. Python modelini çalıştır
params = loadmat('../CURRENT/code/v4_two_phase/sandstone_25C/best_fit_params_Sandstone_25C.mat')
model = AnaerobicModelV4(params['best_params'].flatten(), T=25)

y0 = np.zeros(14)  # başlangıç koşulları
sol = model.solve((0, data_exp['Time'].max()), y0, t_eval=data_exp['Time'].values)

# 3. Karşılaştırma
H2_python = sol.y[0, :]  # Python sonucu
H2_exp = data_exp['H2'].values  # Deneysel

rmse = np.sqrt(np.mean((H2_python - H2_exp)**2))
print(f"Python vs Experimental RMSE: {rmse:.6f}")

# 4. Görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(data_exp['Time'], H2_exp, 'o', label='Experimental')
plt.plot(data_exp['Time'], H2_python, '-', label='Python v4')
plt.xlabel('Time (h)')
plt.ylabel('H2 (mol/L)')
plt.title(f'Sandstone 25°C - RMSE: {rmse:.6f}')
plt.legend()
plt.grid(True)
plt.savefig('HAFTA_3/python_validation.png', dpi=300)

# ✅ Hedef: RMSE < 0.1
if rmse < 0.1:
    print("✅ BAŞARILI! Python modeli doğrulandı.")
else:
    print(f"⚠️ RMSE çok yüksek: {rmse:.6f} > 0.1")
    print("ODE denklemlerini ve parametreleri tekrar kontrol et!")
```

**Çıktı**:
- `HAFTA_3/python_validation.png`
- `HAFTA_3/validation_report.md` (RMSE, farklar, notlar)

---

### 📅  Tüm Kayaçları Python'a Çevir

**Hedef**: 12 durum için Python kodu

```python
# run_all_rocks.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anaerobic_model_v4 import AnaerobicModelV4
from scipy.io import loadmat

rocks = ['Sandstone', 'Basalt', 'Calcite', 'Gypsum']
temps = ['25C', '34C', '40C']

results = {}
rmse_table = []

for rock in rocks:
    for temp in temps:
        rock_lower = rock.lower()

        # Parametreleri yükle
        params_file = f'CURRENT/code/v4_two_phase/{rock_lower}_{temp}/best_fit_params_{rock}_{temp}.mat'
        try:
            params_mat = loadmat(params_file)
            params = params_mat['best_params'].flatten()
        except FileNotFoundError:
            print(f"⚠️ Missing: {params_file}")
            continue

        # Deneysel veriyi yükle
        data_file = f'CURRENT/data/muller_2024/Muller_2024_H2_{rock}_at_{temp}.txt'
        try:
            data_exp = pd.read_csv(data_file, sep='\t')
        except FileNotFoundError:
            print(f"⚠️ Missing: {data_file}")
            continue

        # Modeli çalıştır
        T_celsius = int(temp.replace('C', ''))
        model = AnaerobicModelV4(params, T=T_celsius)

        y0 = np.zeros(14)  # başlangıç koşulları (her kayaç için ayarla)
        sol = model.solve((0, data_exp['Time'].max()), y0, t_eval=data_exp['Time'].values)

        # RMSE hesapla
        H2_python = sol.y[0, :]
        H2_exp = data_exp['H2'].values
        rmse = np.sqrt(np.mean((H2_python - H2_exp)**2))

        # Kaydet
        results[f'{rock}_{temp}'] = {
            'time': data_exp['Time'].values,
            'H2_exp': H2_exp,
            'H2_python': H2_python,
            'rmse': rmse
        }

        rmse_table.append({
            'Rock': rock,
            'Temp': temp,
            'RMSE': rmse,
            'Status': '✅' if rmse < 0.1 else '❌'
        })

        print(f"{rock:12s} {temp:4s} - RMSE: {rmse:.6f} {rmse_table[-1]['Status']}")

# RMSE tablosunu kaydet
df_rmse = pd.DataFrame(rmse_table)
df_rmse.to_excel('HAFTA_5/rmse_summary.xlsx', index=False)

# Tüm sonuçları görselleştir
fig, axes = plt.subplots(4, 3, figsize=(15, 20))
for i, rock in enumerate(rocks):
    for j, temp in enumerate(temps):
        key = f'{rock}_{temp}'
        if key in results:
            ax = axes[i, j]
            r = results[key]
            ax.plot(r['time'], r['H2_exp'], 'o', label='Exp', markersize=3)
            ax.plot(r['time'], r['H2_python'], '-', label='Python')
            ax.set_title(f"{rock} - {temp} (RMSE: {r['rmse']:.4f})")
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('H2 (mol/L)')
            ax.legend()
            ax.grid(True)

plt.tight_layout()
plt.savefig('HAFTA_5/all_rocks_comparison.png', dpi=300)

print("\n📊 Özet:")
print(f"Toplam durum: {len(results)}/12")
print(f"Başarılı (RMSE<0.1): {sum(1 for r in rmse_table if r['RMSE'] < 0.1)}")
```

**Çıktı**:
- `HAFTA_5/all_rocks_comparison.png`
- `HAFTA_5/rmse_summary.xlsx`
- `HAFTA_5/performance_report.md`

---

### 📅 PyTorch + GPU Hazırlık

**Hedef**: LSTM/PINN eğitimi için veri hazırlığı

```python
# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class H2ProductionDataset(Dataset):
    """
    PyTorch Dataset for H2 production data
    """
    def __init__(self, rock_type, temperature, sequence_length=50):
        """
        Args:
            rock_type: 'Sandstone', 'Basalt', 'Calcite', 'Gypsum'
            temperature: '25C', '34C', '40C'
            sequence_length: Number of time steps in each sequence
        """
        # ✅ DOĞRU: CURRENT/ klasöründen veri yükle
        data_file = f'CURRENT/data/muller_2024/Muller_2024_H2_{rock_type}_at_{temperature}.txt'

        # ❌ YANLIŞ: archive/ veya reactions/ klasörüne GİTME!
        # "archive/reactions_old/..." KULLANMA!

        self.data = pd.read_csv(data_file, sep='\t')
        self.sequence_length = sequence_length

        # Normalization
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.data_norm = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Sequence input
        x = self.data_norm.iloc[idx:idx+self.sequence_length].values
        # Target (next time step)
        y = self.data_norm.iloc[idx+self.sequence_length].values

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Test
if __name__ == "__main__":
    dataset = H2ProductionDataset('Sandstone', '25C')
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
```

**Çıktı**: `HAFTA_7/dataset.py` + `HAFTA_7/data_preprocessing.md`

---

## 🔍 GÜNLÜK CHECKLİST

Her çalışma seansına başlarken şunları kontrol et:

- [ ] **Klasör kontrolü**: `pwd` (MATLAB) veya `os.getcwd()` (Python) ile konumu doğrula
- [ ] **Git durumu**: `git status` ile değişiklikleri kontrol et
- [ ] **Dosya yolları**: Kodda "archive" veya "reactions" yok
- [ ] **Versiyon kontrolü**: v4 kodunu kullanıyorsun (dosya adında `_v4.m`)
- [ ] **Not alma**: Gün sonunda `HAFTA_X/NOTES.md` dosyasını güncelle
- [ ] **Git commit**: Önemli değişiklikleri commit'le

---

## ⚠️ YAPMA LİSTESİ

### ❌ Bu Klasörlere GİTME:

```bash
# YANLIŞ - BUNLARI KULLANMA:
archive/reactions_old/Matlab codes/rnn_transport_multiguild_uq_v3.m
archive/reactions_old/Pytorch-ipynb/matlab_to_pytorch_complete.py
archive/reactions_old/dataset_for_training_different_rocks_at_25C-34C-40C/
9_12_2025_calisma/
16.12_2025_calisma/
23_12_2025_calisma/
src/matlab/  # Bu yapı artık yok
```

### ❌ Eski Versiyonları Kullanma:

- v1 kodları (tek fazlı, yanlış fizik)
- v2 kodları (tek fazlı, yanlış fizik)
- v3 kodları (tek fazlı + pH, hala yanlış)
- `rnn_transport_multiguild_uq_v3.m` (eski LSTM kodu)
- `trained_LSTM_multiguild.mat` (eski eğitilmiş model)

### ✅ SADECE Bunları Kullan:

```bash
CURRENT/code/v4_two_phase/        # ✅ v4 iki fazlı kodları
CURRENT/data/muller_2024/         # ✅ Doğru deneysel veri
CURRENT/results/fitted_outputs/   # ✅ Profesörün fit sonuçları
shared/datasets/muller_2024/      # ✅ Ortak veri kopyası
```

---

## 📂 Klasör Yapısı

```
CURRENT/
├── code/v4_two_phase/
│   ├── sandstone_25C/   [.m + .mat]
│   ├── sandstone_34C/   [.m + .mat]
│   ├── sandstone_40C/   [.m + .mat]
│   ├── basalt_25C/      [.m + .mat]
│   ├── basalt_34C/      [.m + .mat]
│   ├── basalt_40C/      [.m + .mat]
│   ├── calcite_25C/     [.m + .mat]
│   ├── calcite_34C/     [.m + .mat]
│   ├── calcite_40C/     [.m + .mat]
│   ├── gypsum_25C/      [.m + .mat]
│   ├── gypsum_34C/      [.m + .mat]
│   └── gypsum_40C/      [.m + .mat]
├── data/muller_2024/    [10 .txt dosyası - Calcite 34C ve 40C eksik]
├── results/fitted_outputs/  [.dat + .png]
└── docs/                [3 .docx - Henry yasası, Fe pool, fit açıklamaları]
```

## 🎓 Mesaj


**Son Güncelleme**: 5 Ocak 2026
**Kaynak**: Mail (3 Ocak 2026, 19:18)

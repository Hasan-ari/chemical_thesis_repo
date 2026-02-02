# LSTM Surrogate Model Geliştirme Rehberi

## 1. Proje Hedefi

**Amaç:** Kimya hocasının MATLAB ODE modelini taklit eden bir LSTM/RNN surrogate model geliştirmek.

**Neden LSTM?**
- Zaman serisi verisi (tekrarlayan yapı)
- ODE çözümü hesaplama maliyetli
- Surrogate model hızlı tahmin sağlar
- Uzun vadeli bağımlılıkları yakalayabilir

## 2. Veri Akışı

```
┌─────────────────────────────────────────────────────────────┐
│                    KİMYA HOCASI                             │
│  MATLAB Kodu + Deneysel Veri → Best Fit Parametreler (.mat) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SENTETİK VERİ ÜRETİMİ                    │
│  .mat parametreleri + ODE çözümü → Zaman serisi verisi      │
│  (Python: scipy.integrate.solve_ivp)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LSTM EĞİTİMİ                             │
│  Sentetik veri + PyTorch → Surrogate Model                  │
└─────────────────────────────────────────────────────────────┘
```

## 3. Girdi/Çıktı Yapısı

### ODE Modeli (14 değişken)
```
Girdi zamanı t'de y vektörü:
y = [nH2_g, nCO2_g, nCH4_g, nH2S_g,      # Gaz fazı (4)
     H2_aq, CO2_aq, SO4, FeS,            # Sulu faz (4)
     X, Acetate, HCO3, S_tot,            # Biyolojik (4)
     Lag, Fe_pool]                        # Kinetik (2)
```

### LSTM Modeli
```
Girdi:  Sequence of states [y(t-seq), y(t-seq+1), ..., y(t-1)]
        Shape: (batch_size, seq_len, 14)

Çıktı:  Next state y(t)
        Shape: (batch_size, 14)
```

## 4. Veri Hazırlama

### 4.1. Sentetik Veri Üretimi (Python)

```python
import numpy as np
from scipy.integrate import solve_ivp
import scipy.io as sio

def generate_synthetic_data(mat_file, t_end=19, n_points=2500):
    """
    .mat dosyasından sentetik veri üret
    """
    # Parametreleri yükle
    mat = sio.loadmat(mat_file)
    p_fit = mat['p_fit'].flatten()
    y0 = mat['y0'].flatten()
    env = extract_env(mat['env'])
    
    # Uniform zaman grid'i
    t_eval = np.linspace(0, t_end, n_points)
    
    # ODE çöz
    sol = solve_ivp(
        lambda t, y: model_mixed(t, y, p_fit, env),
        t_span=[0, t_end],
        y0=y0,
        t_eval=t_eval,
        method='BDF',  # Stiff sistem için
        max_step=0.01
    )
    
    return sol.t, sol.y.T  # Shape: (n_points, 14)
```

### 4.2. Normalizasyon

```python
class DataNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        # Sıfır std için koruma
        self.std[self.std < 1e-10] = 1.0
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data_norm):
        return data_norm * self.std + self.mean
```

### 4.3. Sequence Oluşturma

```python
def create_sequences(data, seq_len=100):
    """
    Sliding window ile sequence'lar oluştur
    
    Args:
        data: (n_samples, n_features) array
        seq_len: Pencere uzunluğu
    
    Returns:
        X: (n_sequences, seq_len, n_features)
        y: (n_sequences, n_features)
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
```

### 4.4. Train/Test Split

```python
def train_test_split_timeseries(X, y, train_ratio=0.8):
    """
    Zaman serisi için sıralı split (shuffle yok!)
    """
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test
```

## 5. LSTM Model Mimarisi

### 5.1. Basit LSTM

```python
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, num_layers=2, output_size=14):
        super(SimpleLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Son zaman adımının çıktısı
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output
```

### 5.2. Stacked LSTM (Önerilen)

```python
class StackedLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_sizes=[128, 64], output_size=14):
        super(StackedLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0],
            hidden_size=hidden_sizes[1],
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_sizes[1], output_size)
    
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.dropout(out1)
        out2, _ = self.lstm2(out1)
        last_out = out2[:, -1, :]
        output = self.fc(last_out)
        return output
```

### 5.3. Bidirectional LSTM (Alternatif)

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, output_size=14):
        super(BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Bidirectional: çıktı boyutu 2x
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output
```

## 6. Eğitim Döngüsü

```python
def train_model(model, train_loader, val_loader, epochs=500, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
    
    return train_losses, val_losses
```

## 7. Free-Running (Chain) Doğrulama

### 7.1. Recursive Prediction

```python
def free_running_prediction(model, initial_sequence, n_steps, normalizer, device):
    """
    Teacher forcing olmadan recursive tahmin
    
    Args:
        model: Eğitilmiş LSTM
        initial_sequence: İlk seq_len adım (normalized)
        n_steps: Tahmin edilecek adım sayısı
        normalizer: Denormalizasyon için
    
    Returns:
        predictions: (n_steps, 14) tahminler (denormalized)
    """
    model.eval()
    
    # Başlangıç sequence'ı kopyala
    current_seq = initial_sequence.copy()  # (seq_len, 14)
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Batch dimension ekle
            x = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
            
            # Tahmin
            y_pred = model(x).cpu().numpy().squeeze()  # (14,)
            predictions.append(y_pred)
            
            # Sequence'ı kaydır: eski ilk adımı çıkar, yeni tahmini ekle
            current_seq = np.vstack([current_seq[1:], y_pred])
    
    predictions = np.array(predictions)
    
    # Denormalize
    predictions_denorm = normalizer.inverse_transform(predictions)
    
    return predictions_denorm
```

### 7.2. Hata Analizi

```python
def analyze_divergence(predictions, ground_truth, checkpoints=[50, 100, 150]):
    """
    Belirli adımlarda hata analizi
    """
    results = {}
    
    for cp in checkpoints:
        if cp <= len(predictions):
            pred_cp = predictions[:cp]
            true_cp = ground_truth[:cp]
            
            # MSE
            mse = np.mean((pred_cp - true_cp) ** 2)
            
            # RMSE
            rmse = np.sqrt(mse)
            
            # MAE
            mae = np.mean(np.abs(pred_cp - true_cp))
            
            # Her değişken için ayrı RMSE
            rmse_per_var = np.sqrt(np.mean((pred_cp - true_cp) ** 2, axis=0))
            
            results[cp] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'rmse_per_variable': rmse_per_var
            }
    
    return results
```

## 8. Değişken İsimleri Referansı

```python
VARIABLE_NAMES = [
    'nH2_g',    # 0: Gaz H2 (mmol)
    'nCO2_g',   # 1: Gaz CO2 (mmol)
    'nCH4_g',   # 2: Gaz CH4 (mmol)
    'nH2S_g',   # 3: Gaz H2S (mmol)
    'H2_aq',    # 4: Çözünmüş H2 (mmol/L)
    'CO2_aq',   # 5: Çözünmüş CO2 (mmol/L)
    'SO4',      # 6: Sülfat (mmol/L)
    'FeS',      # 7: Demir sülfür (mmol/L)
    'X',        # 8: Biyokütle (mmol/L)
    'Acetate',  # 9: Asetat (mmol/L)
    'HCO3',     # 10: Bikarbonat (mmol/L) - SABİT
    'S_tot',    # 11: Toplam sülfür (mmol/L)
    'Lag',      # 12: Lag aktivasyonu (0-1)
    'Fe_pool'   # 13: Fe havuzu (mmol/L)
]

# Değişken grupları
GAS_PHASE = [0, 1, 2, 3]
AQUEOUS_PHASE = [4, 5, 6, 7]
BIOLOGICAL = [8, 9, 10, 11]
KINETIC = [12, 13]
```

## 9. Hiperparametre Önerileri

| Parametre | Önerilen Değer | Notlar |
|-----------|----------------|--------|
| seq_len | 100 | Uzun vadeli bağımlılıklar için |
| hidden_size | 128, 64 (stacked) | İki katmanlı |
| num_layers | 2 | Daha fazlası overfitting riski |
| dropout | 0.2 | Regularizasyon |
| learning_rate | 0.001 | Adam optimizer |
| batch_size | 32 | GPU memory'ye göre ayarla |
| epochs | 500+ | Early stopping ile |
| gradient_clip | 1.0 | Exploding gradient önleme |

## 10. Dikkat Edilecek Noktalar

### 10.1. Veri Kalitesi
- [ ] Tüm değişkenler pozitif olmalı (fiziksel anlam)
- [ ] HCO3 sabit kalmalı (y[:,10] değişmemeli)
- [ ] Lag 0-1 arasında olmalı

### 10.2. Normalizasyon
- [ ] Her değişken için ayrı mean/std
- [ ] Test verisini train istatistikleriyle normalize et
- [ ] Tahminleri denormalize etmeyi unutma

### 10.3. Doğrulama
- [ ] Teacher forcing KULLANMA (free-running doğrulama)
- [ ] Uzun vadeli tahminlerde hata birikimini izle
- [ ] Fiziksel tutarlılık kontrol et (kütle dengesi vb.)

### 10.4. Çoklu Koşullar
- [ ] Her mineral/sıcaklık için ayrı model mi?
- [ ] Yoksa tek model tüm koşullar için mi?
- [ ] Koşul bilgisini girdi olarak ekle (conditional LSTM)

## 11. Dosya Yapısı Önerisi

```
lstm_project/
├── data/
│   ├── raw/                    # .mat ve .txt dosyaları
│   ├── processed/              # Normalize edilmiş veriler
│   └── splits/                 # Train/test split'ler
│
├── models/
│   ├── lstm.py                 # Model tanımları
│   └── saved/                  # Checkpoint'ler
│
├── utils/
│   ├── data_loader.py          # Veri yükleme
│   ├── normalizer.py           # Normalizasyon
│   └── ode_model.py            # Python ODE portu
│
├── train.py                    # Eğitim scripti
├── evaluate.py                 # Değerlendirme
└── config.yaml                 # Hiperparametreler
```

## 12. Sonraki Adımlar

1. **Veri Hazırlama**
   - [ ] Tüm 12 koşul için sentetik veri üret
   - [ ] Normalize et ve sequence'lara böl
   - [ ] Train/test split

2. **Model Geliştirme**
   - [ ] Basit LSTM ile başla
   - [ ] Stacked LSTM dene
   - [ ] Hiperparametre tuning

3. **Doğrulama**
   - [ ] Free-running prediction
   - [ ] Hata analizi (50, 100, 150 adım)
   - [ ] ODE çözümü ile karşılaştırma

4. **İyileştirme**
   - [ ] Attention mekanizması?
   - [ ] Physics-informed loss?
   - [ ] Ensemble modeller?

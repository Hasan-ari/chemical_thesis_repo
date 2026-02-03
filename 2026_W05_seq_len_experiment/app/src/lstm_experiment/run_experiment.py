"""
================================================================================
SEQUENCE LENGTH THRESHOLD EXPERIMENT
================================================================================

DENEY AMACI (EXPERIMENT GOAL):
------------------------------
CS hocamızın istediği: LSTM modelinde window_size (seq_len) parametresini
düşürerek, overfit olmadan minimum değeri bulmak.

Deney tasarımı:
    seq_len = 50 → 30 → 20 → 10 → 5

Her seq_len için:
    1. LSTM'i overfit edene kadar eğit (target_loss = 1e-8)
    2. Autoregressive olarak trajectory üret
    3. RMSE hesapla (ground truth vs generated)
    4. Trajectory "collapse" oldu mu kontrol et

SEQ_LEN NEDİR? (WHAT IS SEQ_LEN?):
----------------------------------
seq_len = "sequence length" = pencere boyutu

Örnek (seq_len=50 için):
    Veri: [y[0], y[1], y[2], ..., y[499]]  (500 nokta, her biri 14 feature)

    Training pair oluşturma:
        Input:  [y[0], y[1], ..., y[49]]   → Output: y[50]
        Input:  [y[1], y[2], ..., y[50]]   → Output: y[51]
        ...
        Input:  [y[449], ..., y[498]]      → Output: y[499]

    Toplam training sample: 500 - 50 = 450 adet

    Her input: (50, 14) shape'inde tensor
    Her output: (14,) shape'inde tensor

SEQ_LEN KÜÇÜLÜNCE NE OLUR?
--------------------------
    seq_len=50: 450 sample, 50 adımlık context
    seq_len=30: 470 sample, 30 adımlık context
    seq_len=20: 480 sample, 20 adımlık context
    seq_len=10: 490 sample, 10 adımlık context
    seq_len=5:  495 sample, 5 adımlık context

    - Daha fazla training sample (iyi)
    - Daha az context (kötü - model trend'i göremez)
    - Bir noktada model öğrenemez → overfit başarısız

BAŞARI KRİTERLERİ:
------------------
    1. Training loss < 1e-8 (overfit başarılı)
    2. Trajectory RMSE < 0.5 (makul tahmin)
    3. Trajectory collapse yok (NaN, Inf, aşırı değerler yok)

================================================================================
Author: Chemical Thesis Project (W05)
Date: 2026-W05
Framework: PyTorch
================================================================================
"""

import json
import logging
import pickle
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ==============================================================================
# SABİT DEĞERLER (CONSTANTS)
# ==============================================================================

# 14 adet durum değişkeni - ODE modelinden gelen state'ler
# Her timestep'te bu 14 değer ölçülüyor/hesaplanıyor
STATE_NAMES: List[str] = [
    "nH2_g",    # 0:  Gaz fazı H2 miktarı (mmol)
    "nCO2_g",   # 1:  Gaz fazı CO2 miktarı (mmol)
    "nCH4_g",   # 2:  Gaz fazı CH4 miktarı (mmol) - metanojenez ürünü
    "nH2S_g",   # 3:  Gaz fazı H2S miktarı (mmol) - sülfür ürünü
    "H2_aq",    # 4:  Çözünmüş H2 konsantrasyonu (mmol/L)
    "CO2_aq",   # 5:  Çözünmüş CO2 konsantrasyonu (mmol/L)
    "SO4",      # 6:  Sülfat konsantrasyonu (mmol/L)
    "FeS",      # 7:  Demir sülfür çökeltisi (mmol/L)
    "X",        # 8:  Biyokütle (bakteriler) (mmol/L)
    "Acetate",  # 9:  Asetat konsantrasyonu (mmol/L)
    "HCO3",     # 10: Bikarbonat (sabit tutulur) (mmol/L)
    "S_tot",    # 11: Toplam çözünmüş sülfür (mmol/L)
    "Lag",      # 12: Lag fazı aktivasyonu (0-1 arası)
    "Fe_pool"   # 13: Kullanılabilir demir havuzu (mmol/L)
]


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Logging sistemi kurulumu.

    Args:
        log_file: Optional - log dosyası yolu
                  Path nesnesi veya None

    Returns:
        logger: logging.Logger nesnesi
                Hem stdout'a hem dosyaya (varsa) yazar

    Örnek kullanım:
        logger = setup_logging(Path("outputs/log.txt"))
        logger.info("Eğitim başladı")
        logger.warning("Dikkat: Yüksek loss!")
    """
    # Logger oluştur - isim vererek diğer modüllerden de erişilebilir
    logger: logging.Logger = logging.getLogger("seq_len_experiment")
    logger.setLevel(logging.INFO)  # INFO ve üstü seviyeleri logla

    # Önceki handler'ları temizle (tekrar çağrılırsa çift log olmasın)
    logger.handlers.clear()

    # Konsola yazan handler
    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(console_handler)

    # Dosyaya yazan handler (opsiyonel)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_handler.formatter)
        logger.addHandler(file_handler)

    return logger


# ==============================================================================
# KONFIGÜRASYON (CONFIGURATION)
# ==============================================================================
@dataclass
class ExperimentConfig:
    """
    Deney konfigürasyonu - tüm hiperparametreler burada.

    @dataclass decorator'ı otomatik olarak:
        - __init__ methodu oluşturur
        - __repr__ methodu oluşturur (print için)
        - Type hint'leri korur

    Attributes:
        data_path (str): Veri dosyasının yolu
            - Format: .npy (NumPy array)
            - Shape: (500, 14) - 500 timestep, 14 feature

        output_dir (str): Çıktıların kaydedileceği klasör

        seq_lengths (List[int]): Test edilecek pencere boyutları
            - Default: [50, 30, 20, 10, 5]
            - Her biri ayrı bir deney olarak çalışır

        n_features (int): Feature sayısı = 14 (state değişkenleri)

        hidden_size (int): LSTM hidden layer boyutu
            - Büyük = daha fazla kapasite, daha yavaş
            - Default: 128 (W04'ten)

        num_layers (int): LSTM katman sayısı
            - Default: 2 (stacked LSTM)

        epochs (int): Maksimum eğitim epoch sayısı
            - Overfit hedeflendiği için yüksek: 10000

        learning_rate (float): Adam optimizer learning rate
            - Default: 5e-4 (0.0005)

        target_loss (float): Hedef loss değeri
            - Bu değerin altına düşünce eğitimi durdur
            - Default: 1e-8 (çok düşük = strong overfit)

        use_log_transform (bool): Log transform uygulansın mı?
            - Bazı değişkenler çok küçük değerler alıyor
            - Log transform onları normalize etmeye yardımcı

        log_cols (tuple): Log transform uygulanacak sütunlar
            - (3, 7, 9, 12, 13) = nH2S_g, FeS, Acetate, Lag, Fe_pool
            - Bunlar küçük veya sıfıra yakın değerler içeriyor

        rmse_threshold (float): Başarı için max RMSE
            - Generated trajectory bu threshold'un altındaysa başarılı

        device (str): PyTorch device
            - "auto": GPU varsa GPU, yoksa CPU
            - "cuda": NVIDIA GPU
            - "mps": Apple Silicon GPU
            - "cpu": CPU

        seed (int): Random seed (reproducibility için)
    """

    # Veri yolları
    data_path: str = "data/basalt_25c_lstm_input_500pts.npy"
    output_dir: str = "outputs/seq_len_experiment"

    # Deney parametreleri - TEST EDİLECEK DEĞERLER
    seq_lengths: List[int] = field(default_factory=lambda: [50, 30, 20, 10, 5])

    # Model mimarisi - SABIT (tüm deneyler için aynı)
    n_features: int = 14      # 14 state değişkeni
    hidden_size: int = 128    # LSTM hidden boyutu
    num_layers: int = 2       # Stacked LSTM katman sayısı

    # Eğitim parametreleri
    epochs: int = 10000       # Max epoch (overfit için yüksek)
    learning_rate: float = 5e-4   # Adam lr (0.0005)
    target_loss: float = 1e-8     # Çok düşük = strong overfit hedefi

    # Veri ön işleme
    use_log_transform: bool = True
    log_cols: tuple = (3, 7, 9, 12, 13)  # Küçük değerli sütunlar

    # Başarı kriterleri
    rmse_threshold: float = 0.5  # Max kabul edilebilir RMSE

    # Cihaz ve seed
    device: str = "auto"
    seed: int = 42

    def to_dict(self) -> dict:
        """
        Config'i dictionary'ye çevir (JSON kaydetmek için).

        Returns:
            dict: Tüm field'ları içeren dictionary

        Not:
            tuple → list dönüşümü gerekli (JSON tuple desteklemez)
        """
        d: dict = asdict(self)
        d['log_cols'] = list(d['log_cols'])  # tuple → list
        return d

    def save(self, path: Path) -> None:
        """
        Config'i JSON dosyasına kaydet.

        Args:
            path: Kayıt yolu (Path nesnesi)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ==============================================================================
# VERİ İŞLEME (DATA PROCESSING)
# ==============================================================================
class DataProcessor:
    """
    Veri yükleme, normalizasyon ve pencere oluşturma sınıfı.

    Bu sınıf şu işlemleri yapar:
        1. .npy dosyasından ham veri yükleme
        2. Log transform (opsiyonel)
        3. StandardScaler ile normalizasyon
        4. Sliding window ile training pair'leri oluşturma
        5. Inverse transform (tahminleri orijinal ölçeğe çevirme)

    Attributes:
        config (ExperimentConfig): Konfigürasyon nesnesi
        logger (logging.Logger): Logger nesnesi
        scaler (StandardScaler): Normalizasyon için scaler
            - fit_transform: eğitimde
            - transform: test/inference'da
            - inverse_transform: tahminleri geri çevirirken

    Örnek kullanım:
        processor = DataProcessor(config, logger)
        data_raw = processor.load_data()  # (500, 14) numpy array
        data_norm = processor.preprocess(data_raw)  # normalized
        X, Y = processor.create_windowed_pairs(data_norm, window_size=50)
    """

    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        """
        DataProcessor constructor.

        Args:
            config: ExperimentConfig nesnesi
            logger: logging.Logger nesnesi
        """
        self.config: ExperimentConfig = config
        self.logger: logging.Logger = logger
        self.scaler: Optional[StandardScaler] = None  # fit_transform'da doldurulacak

    def load_data(self) -> np.ndarray:
        """
        Ham veriyi .npy dosyasından yükle.

        Returns:
            np.ndarray: Ham veri
                - Shape: (500, 14)
                - dtype: float64
                - 500 timestep, her biri 14 feature

        Veri kaynağı:
            - ODE solver (MATLAB ode15s) çıktısı
            - 19 gün simülasyon
            - Uniform timestep'e resample edilmiş
            - dt = 19/499 ≈ 0.038 gün ≈ 55 dakika
        """
        data: np.ndarray = np.load(self.config.data_path)
        self.logger.info(f"Loaded data: {data.shape} from {self.config.data_path}")
        return data

    def preprocess(self, data: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """
        Veri ön işleme: log transform + standardizasyon.

        Args:
            data (np.ndarray): Ham veri
                - Shape: (N, 14) where N = timestep sayısı
            fit_scaler (bool): True ise yeni scaler fit et
                - Eğitimde True
                - Test/inference'da False

        Returns:
            np.ndarray: Normalize edilmiş veri
                - Shape: (N, 14) - aynı shape
                - Değerler: yaklaşık mean=0, std=1

        İşlem adımları:
            1. Veriyi kopyala (orijinali bozmamak için)
            2. Log transform: log1p(x) = log(1 + x)
               - Küçük değerleri genişletir
               - Negatif değerleri önler
            3. StandardScaler: (x - mean) / std
               - Her feature için ayrı mean/std
        """
        # Orijinal veriyi bozmamak için copy
        data: np.ndarray = data.copy()

        # Log transform (opsiyonel)
        if self.config.use_log_transform:
            for col in self.config.log_cols:
                # log1p = log(1 + x), x=0 için log(1)=0 verir
                # maximum(x, 0) negatif değerleri önler
                data[:, col] = np.log1p(np.maximum(data[:, col], 0))

        # Standardizasyon
        if fit_scaler:
            # Yeni scaler oluştur ve fit et
            self.scaler = StandardScaler()
            data_norm: np.ndarray = self.scaler.fit_transform(data)
            self.logger.info("Fitted new StandardScaler")
        else:
            # Mevcut scaler'ı kullan (test verisi için)
            if self.scaler is None:
                raise ValueError("Scaler not fitted yet!")
            data_norm: np.ndarray = self.scaler.transform(data)

        return data_norm

    def inverse_preprocess(self, data_norm: np.ndarray) -> np.ndarray:
        """
        Normalize veriyi orijinal ölçeğe geri çevir.

        Args:
            data_norm (np.ndarray): Normalize edilmiş veri
                - Shape: (N, 14)

        Returns:
            np.ndarray: Orijinal ölçekte veri
                - Shape: (N, 14)

        İşlem adımları (preprocessing'in tersi):
            1. inverse_transform: x * std + mean
            2. expm1: exp(x) - 1 (log1p'nin tersi)
        """
        # Scaler inverse transform
        data: np.ndarray = self.scaler.inverse_transform(data_norm)

        # Log transform'u geri al
        if self.config.use_log_transform:
            for col in self.config.log_cols:
                # expm1 = exp(x) - 1, log1p'nin tersi
                data[:, col] = np.expm1(data[:, col])

        return data

    def create_windowed_pairs(
        self,
        data: np.ndarray,
        window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding window ile training pair'leri oluştur.

        Args:
            data (np.ndarray): Normalize edilmiş veri
                - Shape: (N, 14)
            window_size (int): Pencere boyutu (seq_len)
                - Örn: 50, 30, 20, 10, 5

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, Y) tuple'ı
                - X: Input tensorleri
                    - Shape: (N - window_size, window_size, 14)
                    - Örn: window_size=50 → (450, 50, 14)
                - Y: Target tensorleri
                    - Shape: (N - window_size, 14)
                    - Örn: window_size=50 → (450, 14)

        Sliding window nasıl çalışır:
            Veri: [y[0], y[1], y[2], ..., y[499]]

            window_size=50 için:
                X[0] = [y[0], y[1], ..., y[49]]   → Y[0] = y[50]
                X[1] = [y[1], y[2], ..., y[50]]   → Y[1] = y[51]
                X[2] = [y[2], y[3], ..., y[51]]   → Y[2] = y[52]
                ...
                X[449] = [y[449], ..., y[498]]   → Y[449] = y[499]

            Toplam: 500 - 50 = 450 pair

        ASCII görselleştirme:

            Timestep:  0   1   2   3   ... 49  50  51  ... 499
            Veri:      ●   ●   ●   ●   ... ●   ●   ●   ... ●
                       └─────── X[0] ───────┘   │
                                                Y[0]

                           └─────── X[1] ───────┘   │
                                                    Y[1]
        """
        N: int = len(data)  # Toplam timestep sayısı (500)

        X: List[np.ndarray] = []  # Input listesi
        Y: List[np.ndarray] = []  # Target listesi

        # Sliding window loop
        for i in range(N - window_size):
            # Input: i'den i+window_size'a kadar olan pencere
            window: np.ndarray = data[i : i + window_size]  # (window_size, 14)
            X.append(window)

            # Target: pencerenin hemen sonrasındaki nokta
            target: np.ndarray = data[i + window_size]  # (14,)
            Y.append(target)

        # List'leri numpy array'e çevir
        X_array: np.ndarray = np.array(X)  # (N-window_size, window_size, 14)
        Y_array: np.ndarray = np.array(Y)  # (N-window_size, 14)

        self.logger.info(
            f"Created pairs: X={X_array.shape}, Y={Y_array.shape} "
            f"(window={window_size})"
        )

        return X_array, Y_array


# ==============================================================================
# LSTM MODELİ (LSTM MODEL)
# ==============================================================================
class SeqWindowLSTM(nn.Module):
    """
    Sequence-windowed LSTM modeli.

    Bu model bir pencere (window) alıp sonraki timestep'i tahmin eder:
        Input:  [y[t-W+1], y[t-W+2], ..., y[t]]  → (W, 14) tensor
        Output: y[t+1]                            → (14,) tensor

    Mimari:
        ┌─────────────────────────────────────────┐
        │ Input: (batch, window_size, 14)         │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │ LSTM Layer 1: hidden_size=128           │
        │   - Her timestep için hidden state üret │
        │   - Sequence'ı sırayla işle             │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │ LSTM Layer 2: hidden_size=128           │
        │   - Stacked LSTM (2 katman)             │
        └─────────────────┬───────────────────────┘
                          │
                          ▼ (son timestep'in hidden state'i)
        ┌─────────────────────────────────────────┐
        │ Linear: 128 → 14                        │
        │   - Hidden → Output projection          │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │ Output: (batch, 14)                     │
        └─────────────────────────────────────────┘

    Neden LSTM?
        - Zaman serisi verisi için uygun
        - Long-term dependencies yakalayabilir
        - Hidden state "hafıza" görevi görür

    Attributes:
        lstm (nn.LSTM): LSTM katmanları
            - input_size=14: her timestep 14 feature
            - hidden_size=128: hidden state boyutu
            - num_layers=2: stacked LSTM
            - batch_first=True: input shape (batch, seq, features)

        fc (nn.Linear): Fully connected output layer
            - 128 → 14 projection
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 14
    ):
        """
        LSTM model constructor.

        Args:
            input_size (int): Input feature sayısı (14 state variable)
            hidden_size (int): LSTM hidden state boyutu
            num_layers (int): Stacked LSTM katman sayısı
            output_size (int): Output feature sayısı (14 state variable)
        """
        super().__init__()  # nn.Module'ün __init__'ini çağır

        # LSTM katmanları
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=input_size,      # Her timestep 14 değer
            hidden_size=hidden_size,    # Hidden state boyutu
            num_layers=num_layers,      # Kaç katman (stacked)
            batch_first=True            # (batch, seq, features) formatı
        )

        # Output projection layer
        self.fc: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - modelden geçirme.

        Args:
            x (torch.Tensor): Input tensor
                - Shape: (batch_size, window_size, n_features)
                - Örn: (450, 50, 14)
                - dtype: torch.float32

        Returns:
            torch.Tensor: Tahmin edilen sonraki state
                - Shape: (batch_size, n_features)
                - Örn: (450, 14)

        İşlem adımları:
            1. LSTM'den geçir → her timestep için hidden state
            2. Son timestep'in hidden state'ini al
            3. Linear layer ile 14 feature'a project et
        """
        # LSTM forward pass
        # lstm_out shape: (batch, window_size, hidden_size)
        # hidden: tuple of (h_n, c_n) - final hidden states
        lstm_out, _ = self.lstm(x)

        # Son timestep'in çıktısı (pencereyi gördükten sonra)
        # last_out shape: (batch, hidden_size)
        last_out: torch.Tensor = lstm_out[:, -1, :]

        # Output projection
        # output shape: (batch, output_size) = (batch, 14)
        output: torch.Tensor = self.fc(last_out)

        return output


# ==============================================================================
# TEK DENEY ÇALIŞTIRMA (SINGLE EXPERIMENT)
# ==============================================================================
def run_single_experiment(
    seq_len: int,
    data_norm: np.ndarray,
    data_raw: np.ndarray,
    processor: DataProcessor,
    config: ExperimentConfig,
    device: torch.device,
    logger: logging.Logger,
    output_dir: Path
) -> Dict:
    """
    Tek bir seq_len değeri için deney çalıştır.

    Args:
        seq_len (int): Test edilecek pencere boyutu
            - Örn: 50, 30, 20, 10, 5

        data_norm (np.ndarray): Normalize edilmiş veri
            - Shape: (500, 14)

        data_raw (np.ndarray): Ham (orijinal ölçekte) veri
            - Shape: (500, 14)
            - RMSE hesabı için gerekli

        processor (DataProcessor): Veri işleme nesnesi
            - inverse_preprocess için gerekli

        config (ExperimentConfig): Konfigürasyon

        device (torch.device): PyTorch device (cpu/cuda/mps)

        logger (logging.Logger): Logger

        output_dir (Path): Çıktı klasörü

    Returns:
        Dict: Deney sonuçları
            {
                "seq_len": int,           # Pencere boyutu
                "n_samples": int,         # Training sample sayısı
                "final_loss": float,      # Son training loss
                "epochs_trained": int,    # Kaç epoch eğitildi
                "rmse_total": float,      # Toplam RMSE
                "rmse_per_var": List[float],  # Her değişken için RMSE
                "collapsed": bool,        # Trajectory collapse oldu mu
                "success": bool           # Deney başarılı mı
            }

    Deney adımları:
        1. Training pair'leri oluştur
        2. Model oluştur ve eğit
        3. Autoregressive trajectory üret
        4. RMSE hesapla
        5. Collapse kontrolü yap
        6. Sonuçları kaydet
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT: seq_len = {seq_len}")
    logger.info("=" * 70)

    # Bu deney için output klasörü
    exp_dir: Path = output_dir / f"seq_len_{seq_len}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # ADIM 1: Training pair'leri oluştur
    # =========================================================================
    X, Y = processor.create_windowed_pairs(data_norm, seq_len)
    n_samples: int = len(X)
    logger.info(f"Training samples: {n_samples}")

    # NumPy → PyTorch tensor dönüşümü
    X_tensor: torch.Tensor = torch.FloatTensor(X).to(device)  # (n_samples, seq_len, 14)
    Y_tensor: torch.Tensor = torch.FloatTensor(Y).to(device)  # (n_samples, 14)

    # =========================================================================
    # ADIM 2: Model oluştur ve eğit
    # =========================================================================

    # Reproducibility için seed ayarla
    torch.manual_seed(config.seed)

    # Model oluştur
    model: SeqWindowLSTM = SeqWindowLSTM(
        input_size=config.n_features,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.n_features
    ).to(device)

    # Optimizer ve loss function
    optimizer: torch.optim.Adam = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )
    criterion: nn.MSELoss = nn.MSELoss()  # Mean Squared Error

    # Training history
    history: Dict[str, List] = {"loss": [], "epoch": []}
    final_loss: float = float('inf')

    # Training loop
    for epoch in range(config.epochs):
        model.train()  # Training mode

        # Forward pass
        output: torch.Tensor = model(X_tensor)  # (n_samples, 14)
        loss: torch.Tensor = criterion(output, Y_tensor)

        # Backward pass
        optimizer.zero_grad()  # Gradient'leri sıfırla
        loss.backward()        # Gradient hesapla
        optimizer.step()       # Ağırlıkları güncelle

        # Loss kaydet
        loss_val: float = loss.item()
        history["loss"].append(loss_val)
        history["epoch"].append(epoch)

        # Her 500 epoch'ta log
        if epoch % 500 == 0:
            logger.info(f"  Epoch {epoch:5d} - Loss: {loss_val:.2e}")

        # Target loss'a ulaşınca dur
        if loss_val < config.target_loss:
            logger.info(f"  Target loss reached at epoch {epoch}!")
            final_loss = loss_val
            break

        final_loss = loss_val

    logger.info(f"  Final training loss: {final_loss:.2e}")

    # Model kaydet
    torch.save({
        "model_state_dict": model.state_dict(),
        "seq_len": seq_len,
        "final_loss": final_loss,
        "epochs_trained": len(history["loss"])
    }, exp_dir / "model.pt")

    # =========================================================================
    # ADIM 3: Autoregressive trajectory üret
    # =========================================================================
    model.eval()  # Evaluation mode (dropout vs. kapatılır)

    n_steps: int = len(data_raw)  # 500

    # Trajectory dizisi - ilk seq_len nokta ground truth
    trajectory: np.ndarray = np.zeros((n_steps, config.n_features))
    trajectory[:seq_len] = data_norm[:seq_len]  # İlk pencere = ground truth

    # Sliding window başlangıcı
    window: np.ndarray = data_norm[:seq_len].copy()  # (seq_len, 14)

    # Autoregressive generation
    with torch.no_grad():  # Gradient hesaplama (inference için gerekli değil)
        for t in range(seq_len, n_steps):
            # Window'u tensor'e çevir
            window_tensor: torch.Tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            # Shape: (1, seq_len, 14) - batch=1

            # Tahmin
            pred: torch.Tensor = model(window_tensor)  # (1, 14)
            next_state: np.ndarray = pred.squeeze().cpu().numpy()  # (14,)

            # Trajectory'ye ekle
            trajectory[t] = next_state

            # Window'u kaydır: eski ilk noktayı çıkar, yeni tahmini ekle
            window = np.vstack([window[1:], next_state])

    # =========================================================================
    # ADIM 4: Orijinal ölçeğe çevir ve RMSE hesapla
    # =========================================================================

    # Normalize → orijinal ölçek
    trajectory_orig: np.ndarray = processor.inverse_preprocess(trajectory)

    # RMSE hesapla (sadece generated kısım için, ilk seq_len hariç)
    gen_traj: np.ndarray = trajectory_orig[seq_len:]   # Generated part
    gen_truth: np.ndarray = data_raw[seq_len:]          # Ground truth

    # Her değişken için RMSE
    rmse_per_var: np.ndarray = np.sqrt(np.mean((gen_traj - gen_truth) ** 2, axis=0))

    # Toplam RMSE
    rmse_total: float = float(np.sqrt(np.mean((gen_traj - gen_truth) ** 2)))

    logger.info(f"  Trajectory RMSE (generated part): {rmse_total:.6f}")

    # =========================================================================
    # ADIM 5: Collapse kontrolü
    # =========================================================================

    traj_min: float = float(trajectory_orig.min())
    traj_max: float = float(trajectory_orig.max())
    has_nan: bool = bool(np.isnan(trajectory_orig).any())
    has_inf: bool = bool(np.isinf(trajectory_orig).any())

    # Collapse kriterleri:
    #   - NaN veya Inf varsa
    #   - Değerler çok büyük/küçükse (±1e6)
    collapsed: bool = has_nan or has_inf or traj_max > 1e6 or traj_min < -1e6

    if collapsed:
        logger.info(f"  WARNING: Trajectory collapsed!")

    # =========================================================================
    # ADIM 6: Başarı kontrolü ve sonuçları kaydet
    # =========================================================================

    # Başarı kriterleri:
    #   1. Training loss hedefin 10 katından küçük
    #   2. RMSE threshold'un altında
    #   3. Collapse olmadı
    success: bool = (
        (final_loss < config.target_loss * 10) and
        (rmse_total < config.rmse_threshold) and
        not collapsed
    )

    # Sonuç dictionary
    result: Dict = {
        "seq_len": seq_len,
        "n_samples": n_samples,
        "final_loss": float(final_loss),
        "epochs_trained": len(history["loss"]),
        "rmse_total": float(rmse_total),
        "rmse_per_var": rmse_per_var.tolist(),
        "collapsed": collapsed,
        "success": success
    }

    # JSON olarak kaydet
    with open(exp_dir / "result.json", 'w') as f:
        json.dump(result, f, indent=2)

    # Trajectory kaydet (görselleştirme için)
    np.savez(
        exp_dir / "trajectory.npz",
        trajectory_orig=trajectory_orig,
        ground_truth_orig=data_raw
    )

    # Training history kaydet
    with open(exp_dir / "history.json", 'w') as f:
        json.dump(history, f)

    return result


# ==============================================================================
# GÖRSELLEŞTİRME (VISUALIZATION)
# ==============================================================================
def plot_comparison(
    results: List[Dict],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Tüm deneylerin karşılaştırma grafiklerini oluştur.

    Args:
        results: Her deney için sonuç dictionary'leri
        output_dir: Grafiklerin kaydedileceği klasör
        logger: Logger nesnesi

    Oluşturulan grafikler (2x2 subplot):
        1. RMSE vs seq_len (bar chart)
        2. Training loss vs seq_len (bar chart, log scale)
        3. Training samples vs seq_len (bar chart)
        4. Success/Fail summary (text)
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Verileri çıkar
    seq_lens: List[int] = [r["seq_len"] for r in results]
    rmses: List[float] = [r["rmse_total"] for r in results]
    losses: List[float] = [r["final_loss"] for r in results]
    n_samples: List[int] = [r["n_samples"] for r in results]
    successes: List[bool] = [r["success"] for r in results]

    # Plot 1: RMSE vs seq_len
    ax1 = axes[0, 0]
    colors: List[str] = ['green' if s else 'red' for s in successes]
    ax1.bar(range(len(seq_lens)), rmses, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(seq_lens)))
    ax1.set_xticklabels(seq_lens)
    ax1.set_xlabel("Sequence Length (seq_len)")
    ax1.set_ylabel("RMSE (Generated Trajectory)")
    ax1.set_title("Trajectory RMSE vs Sequence Length")
    ax1.axhline(y=0.5, color='orange', linestyle='--', label='Threshold')
    ax1.legend()

    # Plot 2: Training loss vs seq_len
    ax2 = axes[0, 1]
    ax2.bar(range(len(seq_lens)), losses, color='blue', alpha=0.7)
    ax2.set_xticks(range(len(seq_lens)))
    ax2.set_xticklabels(seq_lens)
    ax2.set_xlabel("Sequence Length (seq_len)")
    ax2.set_ylabel("Final Training Loss")
    ax2.set_title("Training Loss vs Sequence Length")
    ax2.set_yscale('log')  # Log scale (loss çok küçük değerler)
    ax2.axhline(y=1e-8, color='orange', linestyle='--', label='Target')
    ax2.legend()

    # Plot 3: Training samples vs seq_len
    ax3 = axes[1, 0]
    ax3.bar(range(len(seq_lens)), n_samples, color='purple', alpha=0.7)
    ax3.set_xticks(range(len(seq_lens)))
    ax3.set_xticklabels(seq_lens)
    ax3.set_xlabel("Sequence Length (seq_len)")
    ax3.set_ylabel("Number of Training Samples")
    ax3.set_title("Training Samples vs Sequence Length")

    # Plot 4: Success summary (text)
    ax4 = axes[1, 1]
    success_text: List[str] = []
    for r in results:
        status: str = "SUCCESS" if r["success"] else "FAILED"
        collapse: str = " (COLLAPSED)" if r["collapsed"] else ""
        success_text.append(f"seq_len={r['seq_len']}: {status}{collapse}")

    ax4.text(0.1, 0.5, "\n".join(success_text), fontsize=12,
             family='monospace', verticalalignment='center')
    ax4.axis('off')
    ax4.set_title("Experiment Summary")

    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150)
    plt.close()

    logger.info(f"Saved comparison plot to {output_dir / 'comparison.png'}")


def plot_trajectories(
    results: List[Dict],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Her deney için trajectory vs ground truth grafikleri.

    Args:
        results: Deney sonuçları
        output_dir: Çıktı klasörü
        logger: Logger

    Oluşturulan grafik:
        - Satırlar: Key variables (nH2_g, nCH4_g, SO4, X)
        - Sütunlar: Farklı seq_len değerleri
        - Her subplot: Ground truth (mavi) vs Generated (kırmızı)
    """

    # Gösterilecek key değişkenler
    key_vars: List[int] = [0, 2, 6, 8]  # nH2_g, nCH4_g, SO4, X
    var_names: List[str] = [STATE_NAMES[i] for i in key_vars]

    n_vars: int = len(key_vars)
    n_exps: int = len(results)

    fig, axes = plt.subplots(n_vars, n_exps, figsize=(4*n_exps, 3*n_vars))

    for j, r in enumerate(results):
        seq_len: int = r["seq_len"]
        exp_dir: Path = output_dir / f"seq_len_{seq_len}"

        # Trajectory yükle
        data = np.load(exp_dir / "trajectory.npz")
        trajectory: np.ndarray = data["trajectory_orig"]
        ground_truth: np.ndarray = data["ground_truth_orig"]

        t: np.ndarray = np.arange(len(trajectory))

        for i, var_idx in enumerate(key_vars):
            # Subplot seç
            if n_exps > 1:
                ax = axes[i, j]
            else:
                ax = axes[i]

            # Ground truth vs Generated
            ax.plot(t, ground_truth[:, var_idx], 'b-', label='Ground Truth', alpha=0.7)
            ax.plot(t, trajectory[:, var_idx], 'r--', label='Generated', alpha=0.7)

            # seq_len sınırını işaretle
            ax.axvline(x=seq_len, color='gray', linestyle=':', alpha=0.5)

            # Labels
            if i == 0:
                ax.set_title(f"seq_len={seq_len}")
            if j == 0:
                ax.set_ylabel(var_names[i])
            if i == n_vars - 1:
                ax.set_xlabel("Time Step")

            # Legend (sadece ilk subplot)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "trajectories.png", dpi=150)
    plt.close()

    logger.info(f"Saved trajectory plot to {output_dir / 'trajectories.png'}")


# ==============================================================================
# ANA FONKSİYON (MAIN)
# ==============================================================================
def main() -> None:
    """
    Ana deney fonksiyonu - tüm seq_len değerlerini test et.

    Çalıştırma:
        python -m lstm_experiment.run_experiment

    Veya argümanlarla:
        python -m lstm_experiment.run_experiment --seq_lengths 50 30 20
    """
    import argparse

    # Command line argümanları
    parser = argparse.ArgumentParser(
        description="Sequence Length Threshold Experiment"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/basalt_25c_lstm_input_500pts.npy",
        help="Veri dosyası yolu (.npy)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/seq_len_experiment",
        help="Çıktı klasörü"
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[50, 30, 20, 10, 5],
        help="Test edilecek seq_len değerleri"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Max epoch sayısı"
    )
    parser.add_argument(
        "--target_loss",
        type=float,
        default=1e-8,
        help="Hedef loss değeri"
    )

    args = parser.parse_args()

    # Config oluştur
    config = ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seq_lengths=args.seq_lengths,
        epochs=args.epochs,
        target_loss=args.target_loss
    )

    # Output klasörü oluştur
    output_dir: Path = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger kur
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger: logging.Logger = setup_logging(output_dir / f"experiment_{timestamp}.log")

    logger.info("=" * 70)
    logger.info("SEQUENCE LENGTH THRESHOLD EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Testing seq_lengths: {config.seq_lengths}")
    logger.info("")

    # Config kaydet
    config.save(output_dir / "config.json")

    # Device seç
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    logger.info(f"Using device: {device}")

    # Seed ayarla
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Veri yükle ve işle
    processor = DataProcessor(config, logger)
    data_raw: np.ndarray = processor.load_data()
    data_norm: np.ndarray = processor.preprocess(data_raw, fit_scaler=True)

    # =========================================================================
    # TÜM DENEYLER
    # =========================================================================
    results: List[Dict] = []

    for seq_len in config.seq_lengths:
        result = run_single_experiment(
            seq_len=seq_len,
            data_norm=data_norm,
            data_raw=data_raw,
            processor=processor,
            config=config,
            device=device,
            logger=logger,
            output_dir=output_dir
        )
        results.append(result)

    # Tüm sonuçları kaydet
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Grafikler oluştur
    plot_comparison(results, output_dir, logger)
    plot_trajectories(results, output_dir, logger)

    # =========================================================================
    # ÖZET
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE - SUMMARY")
    logger.info("=" * 70)

    for r in results:
        status: str = "SUCCESS" if r["success"] else "FAILED"
        logger.info(
            f"seq_len={r['seq_len']:3d}: {status:7s} | "
            f"loss={r['final_loss']:.2e} | "
            f"RMSE={r['rmse_total']:.4f} | "
            f"samples={r['n_samples']}"
        )

    # Minimum başarılı seq_len bul
    successful: List[Dict] = [r for r in results if r["success"]]
    if successful:
        min_successful: int = min(r["seq_len"] for r in successful)
        logger.info("")
        logger.info(f"MINIMUM SUCCESSFUL seq_len: {min_successful}")
    else:
        logger.info("")
        logger.info("WARNING: No successful experiments!")

    logger.info(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

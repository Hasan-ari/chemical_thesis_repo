# LSTM/Deep Learning Self-Study Plan

Tez için gereken DL bilgisini yapılandırılmış şekilde öğrenmek için haftalık plan.
Her haftanın sonunda kısa not tutarak öğrendiklerini pekiştir.

## Hafta 1: Neural Network Temelleri

**Konular:**
- Perceptron, aktivasyon fonksiyonları (ReLU, sigmoid, tanh)
- Forward pass, loss function (MSE, MAE)
- Backpropagation matematiği (chain rule)
- Gradient descent varyantları (SGD, Adam)

**Kaynaklar:**
- 3Blue1Brown "Neural Networks" serisi (YouTube, 4 video)
- Andrej Karpathy "Micrograd" (YouTube, 2.5 saat) — backprop'u sıfırdan kodla
- Stanford CS231n Lecture 3-4 (backprop + optimization)

**Pratik:** Numpy ile basit 2-layer NN yaz, XOR öğrensin

---

## Hafta 2: Sequence Modelleri — RNN

**Konular:**
- Neden zaman serisi için feedforward NN yetersiz?
- RNN mimarisi: hidden state, weight sharing
- Vanishing/exploding gradient problemi
- BPTT (Backpropagation Through Time)

**Kaynaklar:**
- Andrej Karpathy "The Unreasonable Effectiveness of RNNs" (blog post)
- Stanford CS231n Lecture 10 (RNN)
- Colah's Blog "Understanding LSTM Networks"

**Pratik:** PyTorch ile basit RNN, sinüs dalgası tahmin etsin

---

## Hafta 3: LSTM Derinlemesine

**Konular:**
- LSTM cell anatomisi: forget gate, input gate, output gate, cell state
- Gate formülleri:
  - f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     — forget gate
  - i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     — input gate
  - c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  — candidate cell
  - c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t         — cell state update
  - o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     — output gate
  - h_t = o_t ⊙ tanh(c_t)                    — hidden state
- Neden LSTM gradient vanishing'i çözer? (constant error carousel)
- GRU alternatifi (daha basit, benzer performans)

**Kaynaklar:**
- Colah's Blog "Understanding LSTM Networks" (tekrar, bu sefer formüllerle)
- Aladdin Persson "LSTM from Scratch in PyTorch" (YouTube)
- Original paper: Hochreiter & Schmidhuber 1997 (Abstract + Section 1-3)

**Pratik:** PyTorch `nn.LSTM` ile çok değişkenli zaman serisi tahmini

---

## Hafta 4: Surrogate Modelling Kavramı

**Konular:**
- Surrogate model nedir? (Meta-model, emulator, response surface)
- ODE çözümü vs neural network tahmini: trade-off'lar
- Data-driven vs physics-informed yaklaşımlar
- Autoregressive (free-running) prediction ve hata birikimi

**Kaynaklar:**
- "Neural Network Surrogates for ODE Systems" konulu araştırma (Google Scholar)
- Bu tezin ilham kaynağı: `docs/lstm_fluid_reserach.pdf`
- Review paper: Razavi et al. "Review of surrogate modeling in water resources"

**Pratik:** Basit ODE sistemi (Lotka-Volterra) → LSTM ile surrogate model yaz

---

## Hafta 5: Tez Bağlamında Uygulama

**Konular:**
- PHREEQC çıktıları ile LSTM eğitimi
- Multi-variate time series normalizasyonu
- Sequence length seçimi ve etkisi
- Train/validation/test split stratejileri (zaman serisinde)
- Overfitting tespiti ve regularization (dropout, early stopping)

**Kaynaklar:**
- Projdeki deneysel sonuçlar: `archive/2026_W05_*/`
- PyTorch Time Series Forecasting Tutorial (resmi dokümantasyon)

**Pratik:** PHREEQC verisinden ilk LSTM modeli (Issue #6-10 ile paralel)

---

## Hafta 6: Tez Yazımı için Akademik Çerçeve

**Konular:**
- Literature review yazma: DL + geochemistry kesişimi
- Methodology bölümünde LSTM açıklama şablonu
- Sonuçları bilimsel olarak sunma (RMSE tabloları, trajectory plotları)
- Comparison: ODE vs LSTM (accuracy, speed, generalization)

**Kaynaklar:**
- Benzer tezler (Google Scholar: "LSTM surrogate geochemical" veya "neural network ODE surrogate")
- Advisor'larla tartışma

**Çıktı:** Tez Chapter 3 (Methodology) taslağı

---

## Genel İpuçları

1. **Her hafta sonunda 1 sayfa not tut** — ne öğrendin, ne hala karışık?
2. **Kod yazarak öğren** — video izlemek yetmez, her konuyu kodla
3. **Tez bağlantısı kur** — "Bu kavram tezimde nerede kullanılacak?" sorusunu sürekli sor
4. **Formülleri elle yaz** — gate formüllerini kağıda yaz, dimensionları kontrol et

# MATLAB Ortam Kurulumu

## Gerekli Versiyon
- MATLAB R2023a veya sonrası

## Gerekli Toolbox'lar
1. Deep Learning Toolbox
2. Optimization Toolbox
3. Statistics and Machine Learning Toolbox

## Doğrulama
```matlab
ver  % Yüklü toolbox'ları kontrol et
```

## Path Kurulumu
```matlab
addpath(genpath('src/matlab/core'));
addpath(genpath('src/matlab/visualization'));
addpath(genpath('data/training'));
savepath;
```

## Test
```matlab
cd src/matlab/core
rnn_transport_multiguild_uq_v3
```

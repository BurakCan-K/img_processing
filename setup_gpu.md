# GPU Kurulum Rehberi

## Mevcut Durum
- ✅ GPU: NVIDIA GeForce RTX 3060 6GB (Tespit edildi)
- ✅ CUDA: 12.7 (Kurulu)
- ❌ PyTorch: CPU-only versiyonu kurulu (GPU desteği yok)

## GPU Desteği İçin Kurulum

### 1. Mevcut CPU-only PyTorch'u Kaldır
```bash
.venv\Scripts\activate
pip uninstall torch torchvision
```

### 2. CUDA Destekli PyTorch Kur
CUDA 12.4/12.7 için:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Alternatif olarak (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Kurulumu Doğrula
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### 4. GPU ile Eğitim/Değerlendirme Çalıştır
```bash
# Eğitim
python scripts/train.py --category carpet --device cuda

# Değerlendirme
python scripts/evaluate.py --category carpet --model_path <model_path> --device cuda
```

## Notlar
- RTX 3060 6GB CUDA 12.4/12.7 ile uyumludur
- GPU kullanımı eğitimi ve değerlendirmeyi önemli ölçüde hızlandırır
- GPU bellek kullanımını `nvidia-smi` ile izleyebilirsiniz

